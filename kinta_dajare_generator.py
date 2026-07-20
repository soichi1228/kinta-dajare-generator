from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from pykakasi import kakasi
from sudachipy import dictionary
from contextlib import contextmanager
from time import perf_counter
import unicodedata
from datetime import datetime

# ==========================
# 設定
# ==========================

# 生成する短文数
OUTPUT_COUNT = 100

# 高得点作品の項目別最低点
MIN_MEANING_GAP = 18
MIN_IMAGERY = 18
MIN_PUNCH = 14

# 候補文に含めてはいけない表記
FORBIDDEN_CANDIDATE_PATTERN = re.compile(
    r"(金\s*太|きん\s*た|キン\s*タ)"
)

# 使用モデル
MODEL_NAME = "gemini-3.1-flash-lite"

# 第1段階の出力
OUTPUT_FILE = "kouho.txt"

# 第2段階の出力
KANA_OUTPUT_FILE = "kouho_kana.txt"

# 先頭の「ま」を削除した文章
NO_MA_OUTPUT_FILE = "kouho_kana_no_ma.txt"

# 第3段階でGeminiが変換した文章
CONVERTED_OUTPUT_FILE = "kouho_henkan.txt"

# 第1～第3段階の対応関係を保存するTSV
RESULT_OUTPUT_FILE = "kouho_stage3.tsv"

# 第4段階の評価結果を保存するTSV
EVALUATION_OUTPUT_FILE = "kouho_stage4_v2.tsv"

# 第4段階の評価結果を累積保存するTSV
EVALUATION_HISTORY_FILE = "kouho_stage4_history_v2.tsv"

# 80点以上の作品だけを累積保存するTSV
HIGH_SCORE_OUTPUT_FILE = "kouho_stage4_over80_v2.tsv"

# 高得点作品として保存する最低点
HIGH_SCORE_THRESHOLD = 80

# テストモード
TEST_MODE = False

# テスト用入力ファイル
TEST_INPUT_FILE = "kouho3.txt"

# 過去に生成された候補の累積履歴
HISTORY_FILE = "kouho_history.tsv"

# テストモードでも過去履歴との重複を除外するか
USE_HISTORY_IN_TEST_MODE = False

# テストモードの入力内容を履歴へ追加するか
UPDATE_HISTORY_IN_TEST_MODE = True


# ==========================
# Geminiの構造化出力
# ==========================

class ConversionItem(BaseModel):
    index: int = Field(
        description="入力された文章の番号"
    )

    source_kana: str = Field(
        description="変換前のひらがな文字列"
    )

    converted: str = Field(
        description=(
            "読みを変えずに、漢字・カタカナ・句読点を用いて"
            "自然な日本語に変換した文章。"
            "変換できない場合は『変換不可』とする"
        )
    )

    valid: bool = Field(
        description=(
            "元の読みを変更せず、自然な日本語として成立する場合はtrue。"
            "成立しない場合はfalse"
        )
    )


class ConversionResponse(BaseModel):
    items: list[ConversionItem]


class ValidityItem(BaseModel):
    index: int = Field(
        description="入力された候補の番号"
    )

    kinta_valid: bool = Field(
        description=(
            "金太側の文章が、文脈を補うことで"
            "日本語として成立する場合はtrue"
        )
    )

    kinta_reason: str = Field(
        description="金太側の成立性判定理由"
    )

    kintama_valid: bool = Field(
        description=(
            "キンタマ側の文章が、実在する日本語として"
            "意味を理解できる場合はtrue"
        )
    )

    kintama_reason: str = Field(
        description="キンタマ側の成立性判定理由"
    )


class ValidityResponse(BaseModel):
    items: list[ValidityItem]


class HumorItem(BaseModel):
    index: int = Field(
        description="入力された候補の番号"
    )

    meaning_gap: int = Field(
        ge=0,
        le=25,
        description=(
            "金太側とキンタマ側の意味の落差。0～25点"
        )
    )

    imagery: int = Field(
        ge=0,
        le=25,
        description=(
            "キンタマ側の情景の強さ。0～25点"
        )
    )

    punch: int = Field(
        ge=0,
        le=20,
        description=(
            "オチの分かりやすさと強さ。0～20点"
        )
    )

    rhythm: int = Field(
        ge=0,
        le=15,
        description=(
            "語感、短さ、発音時のリズム。0～15点"
        )
    )

    surprise: int = Field(
        ge=0,
        le=10,
        description=(
            "再解釈の意外性。0～10点"
        )
    )

    originality: int = Field(
        ge=0,
        le=5,
        description=(
            "既視感の少なさと発想の独自性。0～5点"
        )
    )

    comment: str = Field(
        description="面白さに関する簡潔な講評"
    )


class HumorResponse(BaseModel):
    items: list[HumorItem]


# ==========================
# 関数
# ==========================

def validate_exact_result_indexes(
    items: list,
    expected_indexes: set[int],
    stage_name: str,
) -> dict[int, object]:
    """
    Geminiの回答indexが、指定されたindex集合と
    完全に一致することを確認する。
    """
    result_by_index: dict[int, object] = {}

    for item in items:
        if item.index in result_by_index:
            raise RuntimeError(
                f"{stage_name}の回答に重複したindexがあります。"
            )

        result_by_index[item.index] = item

    returned_indexes = set(
        result_by_index.keys()
    )

    if returned_indexes != expected_indexes:
        raise RuntimeError(
            f"{stage_name}の回答indexが一致しません。\n"
            f"期待するindex: {sorted(expected_indexes)}\n"
            f"返されたindex: {sorted(returned_indexes)}"
        )

    return result_by_index

def score_humor_with_gemini(
    client: genai.Client,
    original_sentences: list[str],
    converted_sentences: list[str],
    target_indexes: list[int],
) -> HumorResponse:
    """
    第4B段階として、成立性判定に合格した候補だけを
    面白さの観点で採点する。
    """
    input_items = []

    for index in target_indexes:
        original = original_sentences[index - 1]
        converted = converted_sentences[index - 1]

        input_items.append({
            "index": index,
            "kinta_sentence": f"金太、{original}",
            "kintama_sentence": f"キンタマ、{converted}",
        })

    input_json = json.dumps(
        input_items,
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
あなたは、ぎなた読みによる日本語の言葉遊びを審査する
批評家です。

以下の候補は、別の審査によって、
金太側とキンタマ側の両方が日本語として成立すると
確認されています。

この段階では文章の成立性を再判定せず、
言葉遊びとしての面白さだけを採点してください。

【基本方針】

・現実に起こりそうかどうかでは評価しない
・下品な単語であること自体では加点も減点もしない
・意味が明確なナンセンスは高く評価してよい
・表側から裏側へ意味が大きく切り替わる作品を評価する
・短く、聞いた瞬間に映像とオチが浮かぶ作品を評価する
・説明的で長いだけの作品は低く評価する
・表側と裏側の意味がほとんど同じ作品は低く評価する
・擬人化しただけで物理的な情景が弱い作品は低く評価する
・有名な言葉遊びの軽微な言い換えは低く評価する
・好意的な深読みで点数を引き上げてはいけない

【採点項目】

1. meaning_gap：意味の落差、0～25点

・金太側とキンタマ側で意味が大きく変化する
・先頭の「ま」の所属が変わった効果が明確
・表側から裏側への転換が瞬時に理解できる

単に「真夜中」が「夜中」になるような、
意味がほとんど変わらない候補は低得点にしてください。

2. imagery：情景の強さ、0～25点

・キンタマがどうなったかを瞬時に想像できる
・具体的な動作、接触、破損、変形、移動、状態がある
・短い映像として頭に浮かぶ

抽象的な概念や説明だけの場合は低得点にしてください。

3. punch：オチの強さ、0～20点

・聞いた瞬間に笑いどころが分かる
・裏側の文章が強く、記憶に残る
・説明を加えなくてもオチとして成立する

4. rhythm：語感・リズム、0～15点

・声に出したときに歯切れがよい
・不必要に長くない
・「金太」と「ま」の接続が滑らか
・繰り返して読んだときにリズムがよい

5. surprise：意外性、0～10点

・予想しにくい語の区切りになっている
・表側から裏側への変化に驚きがある
・安易な接頭語の削除だけではない

6. originality：独創性、0～5点

・既存の有名な言葉遊びを連想させにくい
・定型的な構造の使い回しではない
・単語や状況の組み合わせに新しさがある

【採点上の注意】

・各項目を独立して採点する
・合計点は出力しない
・各indexについて必ず1件ずつ回答する
・indexを変更しない
・説明文はJSONの外へ出力しない

【採点対象】
{input_json}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=HumorResponse,
        ),
    )

    if not response.text:
        raise RuntimeError(
            "Geminiから面白さ採点を取得できませんでした。"
        )

    try:
        return HumorResponse.model_validate_json(
            response.text
        )
    except Exception as error:
        raise RuntimeError(
            "Geminiの面白さ採点を解析できませんでした。\n"
            f"Geminiの回答:\n{response.text}"
        ) from error

def contains_forbidden_candidate_text(
    text: str,
) -> bool:
    """
    候補文に「金太」「きんた」「キンタ」などが
    含まれているか確認する。

    空白を挟んだ表記も禁止する。
    """
    normalized = unicodedata.normalize(
        "NFKC",
        text,
    )

    return bool(
        FORBIDDEN_CANDIDATE_PATTERN.search(
            normalized
        )
    )
def normalize_reference_text(text: str) -> str:
    """
    基準例との比較用に、空白や句読点を除去する。
    漢字の違いは残すため、「毛が多い」と「怪我多い」は
    別の表現として扱われる。
    """
    normalized = unicodedata.normalize(
        "NFKC",
        text,
    )

    return IGNORED_READING_CHARS.sub(
        "",
        normalized,
    )


# 読みの比較時に無視する句読点・記号
IGNORED_READING_CHARS = re.compile(
    r"[、。！？!?・「」『』（）()［］\[\]【】〈〉《》〔〕"
    r"…‥：:；;，,．.\s]"
)


@contextmanager
def measure_time(process_name: str):
    start_time = perf_counter()

    try:
        yield
    finally:
        elapsed_time = perf_counter() - start_time
        print(f"{process_name}: {elapsed_time:.2f}秒")


def normalize_kana_for_conversion(text: str) -> str:
    """
    Geminiによる再変換の前に、
    読みに影響しない句読点、記号、空白を除去する。
    """
    return IGNORED_READING_CHARS.sub("", text)


def load_candidate_history(
    history_path: Path,
) -> set[str]:
    """
    累積履歴から、過去に生成された候補の正規化読みを取得する。

    履歴ファイルが存在しない場合は空集合を返す。
    """
    if not history_path.exists():
        return set()

    with history_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file,
            delimiter="\t",
        )

        if (
            reader.fieldnames is None
            or "正規化読み" not in reader.fieldnames
        ):
            raise ValueError(
                "候補履歴ファイルの形式が不正です。\n"
                f"対象ファイル: {history_path}"
            )

        return {
            row["正規化読み"].strip()
            for row in reader
            if row.get("正規化読み", "").strip()
        }


def append_candidate_history(
    history_path: Path,
    records: list[tuple[str, str, str, str, str]],
) -> None:
    """
    今回初めて生成された候補を累積履歴へ追記する。

    recordsの各要素：
    生成日時、元の短文、ひらがな、正規化読み、判定
    """
    if not records:
        return

    is_new_file = (
        not history_path.exists()
        or history_path.stat().st_size == 0
    )

    mode = "w" if is_new_file else "a"
    encoding = "utf-8-sig" if is_new_file else "utf-8"

    with history_path.open(
        mode,
        encoding=encoding,
        newline="",
    ) as file:
        writer = csv.writer(
            file,
            delimiter="\t",
        )

        if is_new_file:
            writer.writerow([
                "生成日時",
                "元の短文",
                "ひらがな",
                "正規化読み",
                "判定",
            ])

        writer.writerows(records)


def format_history_for_prompt(
    history_keys: set[str],
) -> str:
    """
    Geminiへ渡す禁止対象の読み一覧を作成する。
    """
    if not history_keys:
        return "なし"

    return "\n".join(
        f"・{reading}"
        for reading in sorted(history_keys)
    )


def katakana_to_hiragana(text: str) -> str:
    """
    カタカナをひらがなへ変換する。
    """
    return "".join(
        chr(ord(char) - 0x60)
        if "ァ" <= char <= "ヶ"
        else char
        for char in text
    )


def normalize_reading_with_sudachi(
    text: str,
    tokenizer,
) -> str:
    """
    SudachiPyで文章の読みを取得し、
    比較用のひらがな文字列へ正規化する。
    """
    readings: list[str] = []

    for token in tokenizer.tokenize(text):
        reading = token.reading_form()

        # 未知語などで読みを取得できない場合は表層形を使う
        if not reading or reading == "*":
            reading = token.surface()

        readings.append(reading)

    hiragana = katakana_to_hiragana("".join(readings))

    return IGNORED_READING_CHARS.sub("", hiragana)


def convert_to_hiragana(
    text: str,
    converter: kakasi,
) -> str:
    """
    漢字・カタカナを含む日本語をひらがなに変換する。
    句読点などはそのまま残す。
    """
    converted_words = converter.convert(text)

    return "".join(
        word["hira"]
        for word in converted_words
    )


def remove_initial_ma(text: str) -> str:
    """
    文頭の「ま」を1文字だけ削除する。

    例：
    まけるな。 -> けるな。
    まもって。 -> もって。
    """
    text = text.strip().lstrip("\ufeff")

    if not text:
        raise ValueError(
            "空の文章が含まれています。"
        )

    if not text.startswith("ま"):
        raise ValueError(
            f"『ま』で始まっていない文章があります: {text}"
        )

    return text[1:]


def readings_match(
    source_kana: str,
    converted_text: str,
    tokenizer,
) -> bool:
    """
    第3段階の入力音と、Geminiによる変換後の文の読みが
    完全に一致するか判定する。
    """
    source_normalized = normalize_reading_with_sudachi(
        source_kana,
        tokenizer,
    )

    converted_normalized = normalize_reading_with_sudachi(
        converted_text,
        tokenizer,
    )

    return source_normalized == converted_normalized


def compare_full_dajare_readings(
    original_sentence: str,
    converted_sentence: str,
    tokenizer,
) -> tuple[str, str, bool]:
    """
    次の2文の読みを作り、完全一致するか機械判定する。

    ・金太、＋元の短文
    ・キンタマ、＋変換後の短文

    「金太」と「キンタマ」は固有の読みとして固定し、
    後続文の読みをSudachiPyで取得する。
    """
    kinta_suffix = normalize_reading_with_sudachi(
        original_sentence,
        tokenizer,
    )
    kinta_reading = "きんた" + kinta_suffix

    if converted_sentence == "変換不可":
        return kinta_reading, "", False

    kintama_suffix = normalize_reading_with_sudachi(
        converted_sentence,
        tokenizer,
    )
    kintama_reading = "きんたま" + kintama_suffix

    return (
        kinta_reading,
        kintama_reading,
        kinta_reading == kintama_reading,
    )


def validate_result_indexes(
    items: list,
    expected_count: int,
    stage_name: str,
) -> dict[int, object]:
    """
    Geminiの構造化出力について、件数・indexの重複・欠落を確認する。
    """
    result_by_index = {
        item.index: item
        for item in items
    }

    if len(result_by_index) != len(items):
        raise RuntimeError(
            f"{stage_name}の回答に重複したindexがあります。"
        )

    expected_indexes = set(
        range(1, expected_count + 1)
    )
    returned_indexes = set(
        result_by_index.keys()
    )

    if expected_indexes != returned_indexes:
        raise RuntimeError(
            f"{stage_name}の回答件数またはindexが一致しません。\n"
            f"期待するindex: {sorted(expected_indexes)}\n"
            f"返されたindex: {sorted(returned_indexes)}"
        )

    return result_by_index


def convert_with_gemini(
    client: genai.Client,
    no_ma_sentences: list[str],
) -> ConversionResponse:
    """
    先頭の「ま」を削除したひらがな文を、
    読みを変えずに自然な日本語へ変換する。
    """
    input_items = [
        {
            "index": index,
            "source_kana": sentence,
        }
        for index, sentence in enumerate(
            no_ma_sentences,
            start=1,
        )
    ]

    input_json = json.dumps(
        input_items,
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
あなたは、日本語の同音異義語と文の区切りを考える専門家です。

以下のJSONには、ひらがなで書かれた短文があります。
各短文を、読み方を一切変更せず、意味の通る自然な日本語へ
変換してください。

【目的】
変換した文章は「キンタマ、」の後ろに続けて使用します。

【厳守する条件】
・ひらがなの読みを追加、削除、変更しない
・濁点や半濁点も変更しない
・長音、促音、小さい「ゃゅょ」も変更しない
・漢字、カタカナ、句読点への変更はよい
・単語の区切りを変更してよい
・文として意味が通るものにする
・できるだけ日常的で、情景を想像できる表現にする
・各入力につき変換結果を1件だけ返す
・入力のindexを変更しない
・入力のsource_kanaを変更しない
・読みを維持したまま自然な文にできない場合は、
  convertedを「変換不可」、validをfalseとする
・説明文は出力しない
・入力中の句読点と空白は読みには含めない
・必要に応じて句読点を追加してよい
・元の単語区切りに拘束されず、別の単語として再解釈する
・入力文から句読点と空白は事前に除去されている
・読みを維持したまま、単語の区切りを自由に変更する
・必要に応じて自然な位置へ句読点を追加する

【変換例】

入力：
けるな

出力：
蹴るな。

入力：
もって

出力：
持って。

入力：
わった

出力：
割った。

入力：
かおにつく

出力：
顔に付く。

入力：
つかんだ

出力：
掴んだ。

入力：
けがおおい

出力：
毛が多い。

【変換対象】
{input_json}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=ConversionResponse,
        ),
    )

    if not response.text:
        raise RuntimeError(
            "Geminiから第3段階の回答を取得できませんでした。"
        )

    try:
        result = ConversionResponse.model_validate_json(
            response.text
        )
    except Exception as error:
        raise RuntimeError(
            "Geminiの第3段階の回答を解析できませんでした。\n"
            f"Geminiの回答:\n{response.text}"
        ) from error

    return result


def evaluate_validity_with_gemini(
    client: genai.Client,
    original_sentences: list[str],
    converted_sentences: list[str],
) -> ValidityResponse:
    """
    第4A段階として、金太側とキンタマ側が
    日本語として成立しているかを判定する。

    ここでは面白さの採点を行わない。
    """
    input_items = []

    for index, (original, converted) in enumerate(
        zip(
            original_sentences,
            converted_sentences,
        ),
        start=1,
    ):
        input_items.append({
            "index": index,
            "kinta_sentence": f"金太、{original}",
            "kintama_sentence": (
                f"キンタマ、{converted}"
                if converted != "変換不可"
                else "変換不可"
            ),
            "conversion_available": (
                converted != "変換不可"
            ),
        })

    input_json = json.dumps(
        input_items,
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
あなたは、日本語の文章成立性を判定する審査員です。

以下の候補について、
金太側とキンタマ側がそれぞれ日本語として成立しているかを
判定してください。

この段階では、面白さ、下品さ、独創性、現実性は
一切評価しないでください。

【金太側の成立条件】

「金太」を人名として扱ってください。

次のいずれかとして意味を解釈できる場合は、
kinta_validをtrueにしてください。

・金太を主語とする文
・金太への命令、依頼、禁止
・金太への質問、返答、呼びかけ
・金太の状態や性質の説明
・伝言、台詞、看板、記録などの一部
・前後に短い状況説明を付ければ成立する表現
・固有名詞や地名を含む表現
・名詞句や短い断定表現

単独では少し特殊でも、
具体的な使用場面を説明できる場合は成立としてください。

ただし、次の場合はfalseにしてください。

・文法的に解釈できない
・存在しない単語を使用している
・単語が無関係に並んでいるだけ
・極端な深読みをしなければ意味を説明できない

【キンタマ側の成立条件】

「キンタマ」を身体の部位を表す名詞として扱ってください。

次を満たす場合はkintama_validをtrueにしてください。

・実在する日本語だけで構成されている
・単語の区切りと文章構造を説明できる
・キンタマがどうなったか、何をされたか、
  どのような状態かを理解できる
・命令、禁止、状態、動作、受け身などとして解釈できる

現実には起こりにくいことでも、
日本語として意味が明確なら成立としてください。

例えば、空を飛ぶ、巨大化する、凍るなど、
非現実的な内容であることだけを理由に
不成立としてはいけません。

ただし、次の場合はfalseにしてください。

・存在しない動詞や名詞を作っている
・助詞や活用が破綻している
・誰が何をしたか全く理解できない
・抽象語を並べただけで具体的な意味がない
・好意的な深読みをしなければ成立しない

【重要】

・conversion_availableがfalseの場合、
  kintama_validは必ずfalseとする
・面白さは採点しない
・現実的かどうかは評価しない
・下品な単語であることは判定に影響させない
・各indexについて必ず1件ずつ回答する
・indexを変更しない
・説明文はJSONの外へ出力しない

【判定対象】
{input_json}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=ValidityResponse,
        ),
    )

    if not response.text:
        raise RuntimeError(
            "Geminiから成立性判定を取得できませんでした。"
        )

    try:
        return ValidityResponse.model_validate_json(
            response.text
        )
    except Exception as error:
        raise RuntimeError(
            "Geminiの成立性判定を解析できませんでした。\n"
            f"Geminiの回答:\n{response.text}"
        ) from error


def write_stage3_tsv(
    output_path: Path,
    original_sentences: list[str],
    kana_sentences: list[str],
    no_ma_sentences: list[str],
    converted_sentences: list[str],
    model_valid_results: list[bool],
    reading_match_results: list[bool],
) -> None:
    """
    第1～第3段階の対応関係をTSVファイルへ保存する。
    """
    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.writer(
            file,
            delimiter="\t",
        )

        writer.writerow([
            "番号",
            "元の短文",
            "ひらがな",
            "先頭のまを削除",
            "Gemini変換結果",
            "Gemini判定",
            "読み一致",
        ])

        for index, values in enumerate(
            zip(
                original_sentences,
                kana_sentences,
                no_ma_sentences,
                converted_sentences,
                model_valid_results,
                reading_match_results,
            ),
            start=1,
        ):
            (
                original,
                kana,
                no_ma,
                converted,
                model_valid,
                reading_match,
            ) = values

            writer.writerow([
                index,
                original,
                kana,
                no_ma,
                converted,
                model_valid,
                reading_match,
            ])


def write_stage4_tsv(
    output_path: Path,
    original_sentences: list[str],
    kana_sentences: list[str],
    no_ma_sentences: list[str],
    converted_sentences: list[str],
    stage3_model_valid_results: list[bool],
    stage3_reading_match_results: list[bool],
    validity_items: list[ValidityItem],
    humor_items: list[HumorItem],
    kinta_readings: list[str],
    kintama_readings: list[str],
    full_reading_match_results: list[bool],
) -> None:
    """
    成立性判定と面白さ採点をTSVへ保存する。
    """
    validity_by_index = {
        item.index: item
        for item in validity_items
    }

    humor_by_index = {
        item.index: item
        for item in humor_items
    }

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.writer(
            file,
            delimiter="\t",
        )

        writer.writerow([
            "番号",
            "元の短文",
            "ひらがな",
            "先頭のまを削除",
            "Gemini変換結果",
            "第3段階Gemini判定",
            "第3段階読み一致",
            "金太側の文",
            "金太側成立",
            "金太側判定理由",
            "キンタマ側の文",
            "キンタマ側成立",
            "キンタマ側判定理由",
            "作品成立",
            "意味の落差_25点",
            "情景の強さ_25点",
            "オチの強さ_20点",
            "語感・リズム_15点",
            "意外性_10点",
            "独創性_5点",
            "合計_100点",
            "面白さ講評",
            "金太側の読み",
            "キンタマ側の読み",
            "両文の読み完全一致",
        ])

        for index, values in enumerate(
            zip(
                original_sentences,
                kana_sentences,
                no_ma_sentences,
                converted_sentences,
                stage3_model_valid_results,
                stage3_reading_match_results,
                kinta_readings,
                kintama_readings,
                full_reading_match_results,
            ),
            start=1,
        ):
            (
                original,
                kana,
                no_ma,
                converted,
                stage3_model_valid,
                stage3_reading_match,
                kinta_reading,
                kintama_reading,
                full_reading_match,
            ) = values

            validity = validity_by_index[index]

            work_valid = (
                converted != "変換不可"
                and full_reading_match
                and validity.kinta_valid
                and validity.kintama_valid
            )

            if work_valid:
                humor = humor_by_index.get(
                    index
                )

                if humor is None:
                    raise RuntimeError(
                        "成立作品の面白さ採点がありません。\n"
                        f"index: {index}"
                    )

                meaning_gap = humor.meaning_gap
                imagery = humor.imagery
                punch = humor.punch
                rhythm = humor.rhythm
                surprise = humor.surprise
                originality = humor.originality
                humor_comment = humor.comment

            else:
                meaning_gap = 0
                imagery = 0
                punch = 0
                rhythm = 0
                surprise = 0
                originality = 0

                humor_comment = (
                    "成立性判定または読み一致判定で"
                    "不合格となったため採点対象外。"
                )

            total_score = (
                meaning_gap
                + imagery
                + punch
                + rhythm
                + surprise
                + originality
            )

            kinta_sentence = (
                f"金太、{original}"
            )

            kintama_sentence = (
                f"キンタマ、{converted}"
                if converted != "変換不可"
                else "変換不可"
            )

            writer.writerow([
                index,
                original,
                kana,
                no_ma,
                converted,
                stage3_model_valid,
                stage3_reading_match,
                kinta_sentence,
                validity.kinta_valid,
                validity.kinta_reason,
                kintama_sentence,
                validity.kintama_valid,
                validity.kintama_reason,
                work_valid,
                meaning_gap,
                imagery,
                punch,
                rhythm,
                surprise,
                originality,
                total_score,
                humor_comment,
                kinta_reading,
                kintama_reading,
                full_reading_match,
            ])

def append_stage4_history_tsv(
    latest_output_path: Path,
    history_output_path: Path,
    run_id: str,
    executed_at: str,
    test_mode: bool,
    model_name: str,
) -> int:
    """
    今回作成したkouho_stage4.tsvの内容を、
    累積履歴ファイルへ追記する。

    戻り値は追記したデータ件数。
    """

    if not latest_output_path.exists():
        raise FileNotFoundError(
            "今回の第4段階評価ファイルが見つかりません。\n"
            f"対象ファイル: {latest_output_path}"
        )

    # 今回の評価ファイルを読み込む
    with latest_output_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.reader(
            file,
            delimiter="\t",
        )

        rows = list(reader)

    if not rows:
        raise ValueError(
            "今回の第4段階評価ファイルが空です。\n"
            f"対象ファイル: {latest_output_path}"
        )

    latest_header = rows[0]
    latest_data_rows = rows[1:]

    if not latest_data_rows:
        return 0

    # 累積ファイルには、実行情報を先頭へ追加する
    history_header = [
        "実行ID",
        "実行日時",
        "実行モード",
        "使用モデル",
        *latest_header,
    ]

    is_new_file = (
        not history_output_path.exists()
        or history_output_path.stat().st_size == 0
    )

    # 既存の履歴ファイルがある場合は、
    # 列構成が現在のコードと一致するか確認する
    if not is_new_file:
        with history_output_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.reader(
                file,
                delimiter="\t",
            )

            existing_header = next(
                reader,
                None,
            )

        if existing_header != history_header:
            raise ValueError(
                "第4段階の累積履歴ファイルの列構成が、"
                "現在のプログラムと一致しません。\n"
                f"対象ファイル: {history_output_path}\n"
                "既存ファイルを別名へ変更するか、削除してください。"
            )

    mode = "w" if is_new_file else "a"
    encoding = (
        "utf-8-sig"
        if is_new_file
        else "utf-8"
    )

    history_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with history_output_path.open(
        mode,
        encoding=encoding,
        newline="",
    ) as file:
        writer = csv.writer(
            file,
            delimiter="\t",
        )

        if is_new_file:
            writer.writerow(
                history_header
            )

        execution_mode = (
            "TEST"
            if test_mode
            else "NORMAL"
        )

        for row in latest_data_rows:
            writer.writerow([
                run_id,
                executed_at,
                execution_mode,
                model_name,
                *row,
            ])

    return len(latest_data_rows)


def parse_tsv_bool(value: str) -> bool:
    """
    TSVへ保存された真偽値をboolへ変換する。
    """
    return value.strip().lower() in {
        "true",
        "1",
        "yes",
        "はい",
        "合格",
    }


def make_work_key(
    original: str,
    converted: str,
) -> str:
    """
    同じ作品を二重登録しないための比較キーを作る。

    句読点や空白の違いは無視し、
    漢字・語句の違いは区別する。
    """
    original_key = normalize_reference_text(
        original
    )
    converted_key = normalize_reference_text(
        converted
    )

    return f"{original_key}|||{converted_key}"


def append_high_score_works_tsv(
    latest_output_path: Path,
    high_score_output_path: Path,
    run_id: str,
    executed_at: str,
    model_name: str,
    threshold: int,
) -> tuple[int, int, int]:
    """
今回の第4段階評価ファイルから、次の条件をすべて満たす作品だけを
高得点作品ファイルへ累積保存する。

・合計点がthreshold以上
・意味の落差がMIN_MEANING_GAP以上
・情景の強さがMIN_IMAGERY以上
・オチの強さがMIN_PUNCH以上
・金太側とキンタマ側の両方が成立
・両文の読みが完全一致
・変換不可ではない
・同じ作品がまだ登録されていない
"""
    if not latest_output_path.exists():
        raise FileNotFoundError(
            "今回の第4段階評価ファイルが見つかりません。\n"
            f"対象ファイル: {latest_output_path}"
        )

    with latest_output_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file,
            delimiter="\t",
        )

        latest_header = reader.fieldnames
        latest_rows = list(reader)

    if latest_header is None:
        raise ValueError(
            "今回の第4段階評価ファイルに見出しがありません。\n"
            f"対象ファイル: {latest_output_path}"
        )

    required_columns = {
        "元の短文",
        "Gemini変換結果",
        "作品成立",
        "意味の落差_25点",
        "情景の強さ_25点",
        "オチの強さ_20点",
        "合計_100点",
        "両文の読み完全一致",
    }

    missing_columns = (
        required_columns
        - set(latest_header)
    )

    if missing_columns:
        raise ValueError(
            "今回の第4段階評価ファイルに必要な列がありません。\n"
            f"不足列: {sorted(missing_columns)}\n"
            f"対象ファイル: {latest_output_path}"
        )

    high_score_header = [
        "作品キー",
        "初回登録実行ID",
        "初回登録日時",
        "使用モデル",
        *latest_header,
    ]

    high_score_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    is_new_file = (
        not high_score_output_path.exists()
        or high_score_output_path.stat().st_size == 0
    )

    existing_keys: set[str] = set()

    if not is_new_file:
        with high_score_output_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(
                file,
                delimiter="\t",
            )

            existing_header = reader.fieldnames

            if existing_header != high_score_header:
                raise ValueError(
                    "高得点作品の累積ファイルの列構成が、"
                    "現在のプログラムと一致しません。\n"
                    f"対象ファイル: {high_score_output_path}\n"
                    "既存ファイルを別名へ変更するか、削除してください。"
                )

            existing_keys = {
                row["作品キー"].strip()
                for row in reader
                if row.get("作品キー", "").strip()
            }

    qualified_rows: list[
        tuple[str, dict[str, str]]
    ] = []

    for row in latest_rows:
        # 総合点を取得
        try:
            total_score = int(
                row["合計_100点"].strip()
            )
        except (
            TypeError,
            ValueError,
            AttributeError,
        ):
            total_score = 0

        # 項目別得点を取得
        try:
            meaning_gap = int(
                row["意味の落差_25点"].strip()
            )
            imagery = int(
                row["情景の強さ_25点"].strip()
            )
            punch = int(
                row["オチの強さ_20点"].strip()
            )
        except (
            TypeError,
            ValueError,
            AttributeError,
        ):
            meaning_gap = 0
            imagery = 0
            punch = 0

        converted = row[
            "Gemini変換結果"
        ].strip()

        work_valid = parse_tsv_bool(
            row["作品成立"]
        )

        full_reading_match = parse_tsv_bool(
            row["両文の読み完全一致"]
        )

        # 総合点による足切り
        if total_score < threshold:
            continue

        # 項目別得点による足切り
        if meaning_gap < MIN_MEANING_GAP:
            continue

        if imagery < MIN_IMAGERY:
            continue

        if punch < MIN_PUNCH:
            continue

        # 作品として成立していないものは除外
        if not work_valid:
            continue

        # 読みが完全一致しないものは除外
        if not full_reading_match:
            continue

        # 変換不可は除外
        if converted == "変換不可":
            continue

        work_key = make_work_key(
            row["元の短文"],
            converted,
        )

        qualified_rows.append((
            work_key,
            row,
        ))

    new_rows: list[
        tuple[str, dict[str, str]]
    ] = []

    # 同じ実行内で同じ作品が複数あった場合も1件にする
    current_keys: set[str] = set()

    for work_key, row in qualified_rows:
        if work_key in existing_keys:
            continue

        if work_key in current_keys:
            continue

        current_keys.add(
            work_key
        )
        new_rows.append((
            work_key,
            row,
        ))

    duplicate_count = (
        len(qualified_rows)
        - len(new_rows)
    )

    if not new_rows:
        # 初回実行で該当作品が0件でも、
        # 見出しだけの累積ファイルを作成する。
        if is_new_file:
            with high_score_output_path.open(
                "w",
                encoding="utf-8-sig",
                newline="",
            ) as file:
                writer = csv.writer(
                    file,
                    delimiter="\t",
                )
                writer.writerow(
                    high_score_header
                )

        return (
            len(qualified_rows),
            0,
            duplicate_count,
        )

    mode = "w" if is_new_file else "a"
    encoding = (
        "utf-8-sig"
        if is_new_file
        else "utf-8"
    )

    with high_score_output_path.open(
        mode,
        encoding=encoding,
        newline="",
    ) as file:
        writer = csv.writer(
            file,
            delimiter="\t",
        )

        if is_new_file:
            writer.writerow(
                high_score_header
            )

        for work_key, row in new_rows:
            writer.writerow([
                work_key,
                run_id,
                executed_at,
                model_name,
                *[
                    row.get(column, "")
                    for column in latest_header
                ],
            ])

    return (
        len(qualified_rows),
        len(new_rows),
        duplicate_count,
    )


# ==========================
# メイン処理
# ==========================

def main() -> None:
    # このpyファイルが保存されているフォルダ
    script_dir = Path(__file__).resolve().parent

    output_path = script_dir / OUTPUT_FILE
    kana_output_path = script_dir / KANA_OUTPUT_FILE
    no_ma_output_path = script_dir / NO_MA_OUTPUT_FILE
    converted_output_path = script_dir / CONVERTED_OUTPUT_FILE
    result_output_path = script_dir / RESULT_OUTPUT_FILE
    evaluation_output_path = script_dir / EVALUATION_OUTPUT_FILE
    evaluation_history_output_path = (
        script_dir
        / EVALUATION_HISTORY_FILE
    )
    high_score_output_path = (
        script_dir
        / HIGH_SCORE_OUTPUT_FILE
    )
    test_input_path = script_dir / TEST_INPUT_FILE
    history_path = script_dir / HISTORY_FILE

    # ひらがな変換器
    converter = kakasi()

    # SudachiPyのトークナイザー
    sudachi_tokenizer = dictionary.Dictionary().create()

    # Geminiクライアントを作成
    client = genai.Client()

    # 通常モードでは累積履歴を使用する。
    # テストモードでは設定に応じて履歴を使用する。
    if TEST_MODE and not USE_HISTORY_IN_TEST_MODE:
        history_keys: set[str] = set()
    else:
        history_keys = load_candidate_history(
            history_path
        )

    try:
        # ==========================================
        # 第1段階：候補文の生成またはテスト文の読込み
        # ==========================================

        if TEST_MODE:
            if not test_input_path.exists():
                raise FileNotFoundError(
                    f"テスト用ファイルがありません: {test_input_path}"
                )

            original_sentences = [
                line.strip()
                for line in test_input_path.read_text(
                    encoding="utf-8-sig"
                ).splitlines()
                if line.strip()
            ]

            if not original_sentences:
                raise ValueError(
                    f"{TEST_INPUT_FILE}にテスト例文がありません。"
                )

            candidate_text = "\n".join(
                original_sentences
            )

        else:
            excluded_readings_text = format_history_for_prompt(
                history_keys
            )

            prompt = f"""
あなたは日本語の短文生成AIです。

以下の条件を満たす短文を{OUTPUT_COUNT}件作成してください。

【最重要条件】
・各文をひらがなにしたとき、最初の1文字が必ず「ま」になること
・「み」「む」「め」「も」で始まる文は禁止
・出力前に、各文の読みの最初の1文字が「ま」であることを確認すること
・3~13文字程度であること

【出力する文章の範囲】
・出力するのは、後から文頭に人物名を付けられる「後続部分」だけとする
・出力文の中に、主語となる人物名を入れない
・出力文の末尾に人物名を付け足さない
・「金太」「きんた」「キンタ」という文字列を一切含めない
・人物名を省略した状態でも、動作、状態、命令、質問などとして
  意味を解釈できる文章にする
・人物名を文頭、文中、文末のいずれにも出力しない


【文字数内訳の厳守（まんべんなく出力するためのノルマ）】
出力する {OUTPUT_COUNT} 件の文字数を意図的にバラバラに散らばらせてください。長い文ばかりに偏るのは絶対に禁止です。以下の内訳を目標に作成してください。

1. 【超短文枠（3〜5文字程度）】: 約35件
   （動詞単体や、極めて短いフレーズ）
   ・例：「守る」「待った」「満足だ」

2. 【短文枠（6〜8文字程度）】: 約35件
   （名詞＋動詞などのシンプルな構造）
   ・例：「的を射る」「マントを着る」「漫画を読む」

3. 【中長文枠（9〜13文字程度）】: 約30件
   （少し状況を描写した文）
   ・例：「満員のバスから降りる」
   

【その他の条件】
・文法的に自然
・できるだけ内容が重複しない
・番号は付けない
・箇条書き記号は付けない
・1行につき1文だけ出力する
・説明は不要
・通常の主語述語文だけに限定しない
・命令、依頼、禁止、質問、返答、呼びかけ、状態の説明、
  伝言、台詞、看板、名詞句なども使用してよい
・単独では多少特殊でも、前後に短い状況説明を付ければ
  日本語として成立する表現は使用してよい
・現実に起こりやすい内容である必要はない
・ただし、意味を説明できない単語の羅列や、
  文法的に解釈できない文字列は禁止
・抽象的な概念だけの文章より、
  人、身体、物、場所、動作、状態が含まれる表現を優先する

【カテゴリー構成】
出力内容が特定の種類に偏らないよう、
次のカテゴリーをおおむね指定された割合で作成してください。

1. 命令・依頼・禁止：約15％
2. 人や物の動作・行為：約15％
3. 状態・性質・変化：約15％
4. 質問・返答・呼びかけ・感嘆：約10％
5. 地名・施設名・店名・商品名など、人名以外の固有名詞を含む文：約10％
6. 伝言・台詞・看板・記録などの文：約10％
7. 道具・食物・生物・身体・物体を含む文：約10％
8. 事故・異変・失敗・突発的な出来事：約10％
9. 名詞句・短い断定・短い状態表現：約5％

各文は主となるカテゴリーを1つ持つようにしてください。
同じ構文、同じ語尾、同じ動詞、同じ接頭語を
何度も繰り返さないでください。

【過去に生成済みの読みの禁止】
次の一覧は、過去に生成された短文を、
ひらがなに変換して句読点と空白を除去したものです。

今回生成する短文を同じ方法でひらがなに変換した結果が、
次の一覧のいずれかと一致する場合は出力しないでください。

{excluded_readings_text}
"""

            with measure_time("第1段階：Gemini候補生成"):
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
            )

            if not response.text:
                raise RuntimeError(
                    "Geminiから候補文を取得できませんでした。"
                )

            original_sentences = [
                line.strip()
                for line in response.text.splitlines()
                if line.strip()
            ]

            candidate_text = "\n".join(
                original_sentences
            )

            output_path.write_text(
                candidate_text,
                encoding="utf-8",
            )

        # ==========================================
        # 第2段階：ひらがなへの変換
        # ==========================================

        with measure_time("第2段階：ひらがな変換"):
            raw_kana_sentences = [
                convert_to_hiragana(
                    sentence,
                    converter,
                )
                for sentence in original_sentences
            ]


        # ==========================================
        # 「ま」始まりと過去履歴・今回内の重複を検査する
        # ==========================================

        valid_original_sentences: list[str] = []
        valid_kana_sentences: list[str] = []
        rejected_sentences: list[tuple[str, str]] = []
        duplicate_sentences: list[tuple[str, str, str]] = []
        forbidden_sentences: list[tuple[str, str]] = []

        # 今回の回答内で同じ読みが複数回出た場合も除外する
        current_batch_keys: set[str] = set()

        # 通常モードでは履歴へ追記する。
        # テストモードでは設定に応じて追記する。
        update_history = (
            not TEST_MODE
            or UPDATE_HISTORY_IN_TEST_MODE
        )

        generated_at = datetime.now().isoformat(
            timespec="seconds"
        )

        history_records: list[
            tuple[str, str, str, str, str]
        ] = []

        for original, kana in zip(
            original_sentences,
            raw_kana_sentences,
        ):
            # 句読点や空白を除いた読み
            normalized_kana = normalize_kana_for_conversion(
                kana
            )

            # ======================================
            # 候補の判定
            # ======================================

            if contains_forbidden_candidate_text(
                original
            ):
                status = "禁止語を含む"

                forbidden_sentences.append((
                    original,
                    kana,
                ))

            elif normalized_kana in history_keys:
                status = "過去履歴と重複"

                duplicate_sentences.append((
                    original,
                    kana,
                    status,
                ))

            elif normalized_kana in current_batch_keys:
                status = "今回の生成結果内で重複"

                duplicate_sentences.append((
                    original,
                    kana,
                    status,
                ))

            else:
                current_batch_keys.add(
                    normalized_kana
                )

                if normalized_kana.startswith("ま"):
                    status = "採用"

                    valid_original_sentences.append(
                        original
                    )
                    valid_kana_sentences.append(
                        kana
                    )

                else:
                    status = "読みが「ま」以外"

                    rejected_sentences.append((
                        original,
                        kana,
                    ))
            # ======================================
            # Geminiが生成した全候補を履歴へ記録
            # ======================================

            if update_history:
                history_records.append((
                    generated_at,
                    original,
                    kana,
                    normalized_kana,
                    status,
                ))

        # ==========================================
        # 履歴ファイルへ追記
        # ==========================================

        if update_history:
            append_candidate_history(
                history_path,
                history_records,
            )

            print(
                f"今回の履歴追記件数: "
                f"{len(history_records)}件"
            )

            print(
                f"履歴保存先: "
                f"{history_path.resolve()}"
            )

        # 有効な候補だけに差し替える
        original_sentences = valid_original_sentences
        kana_sentences = valid_kana_sentences

        if not original_sentences:
            raise ValueError(
                "新規かつ読みが「ま」で始まる候補が1件もありませんでした。"
            )


        # 有効な候補だけをkouho.txtへ保存
        candidate_text = "\n".join(
            original_sentences
        )

        output_path.write_text(
            candidate_text,
            encoding="utf-8",
        )


        # 有効な候補の読みだけをkouho_kana.txtへ保存
        kana_text = "\n".join(
            kana_sentences
        )

        kana_output_path.write_text(
            kana_text,
            encoding="utf-8",
        )


        # 後続処理で使用
        kana_sentences_from_file = kana_sentences


        print(
            f"Gemini生成件数: {len(raw_kana_sentences)}件"
        )
        print(
            f"新規の「ま」始まり採用件数: "
            f"{len(original_sentences)}件"
        )
        print(
            f"「ま」以外の除外件数: "
            f"{len(rejected_sentences)}件"
        )
        print(
            f"重複による除外件数: "
            f"{len(duplicate_sentences)}件"
        )
        print(f"禁止語による除外件数: "f"{len(forbidden_sentences)}件")

        if rejected_sentences:
            print("\n【読みが「ま」以外の候補】")

            for original, kana in rejected_sentences:
                print(
                    f"{original} → {kana}"
                )

        if duplicate_sentences:
            print("\n【過去または今回の重複候補】")

            for original, kana, reason in duplicate_sentences:
                print(
                    f"{original} → {kana}（{reason}）"
                )

        if update_history:
            print(
                f"候補履歴: {history_path}"
            )

        # ==========================================
        # 第3段階：先頭の「ま」を削除
        # ==========================================

        no_ma_sentences = [
            remove_initial_ma(sentence)
            for sentence in kana_sentences_from_file
        ]

        no_ma_text = "\n".join(
            no_ma_sentences
        )

        # Geminiに渡す際は句読点や空白を除去する
        conversion_input_sentences = [
            normalize_kana_for_conversion(sentence)
            for sentence in no_ma_sentences
        ]

        no_ma_output_path.write_text(
            no_ma_text,
            encoding="utf-8",
        )

        # ==========================================
        # 第3段階：Geminiで自然な日本語へ変換
        # ==========================================

        with measure_time("第3段階：Gemini変換"):
            conversion_result = convert_with_gemini(
                client,
                conversion_input_sentences,
            )

        result_by_index = validate_result_indexes(
            conversion_result.items,
            len(conversion_input_sentences),
            "第3段階",
        )

        converted_sentences: list[str] = []
        model_valid_results: list[bool] = []
        reading_match_results: list[bool] = []

        for index, source_kana in enumerate(
            conversion_input_sentences,
            start=1,
        ):
            item = result_by_index[index]

            # Geminiが返したsource_kanaを正規化する
            returned_source_kana = (
                normalize_kana_for_conversion(
                    item.source_kana
                )
            )

            # Geminiが入力文字列を変更した場合は、
            # プログラム全体を止めず、その候補だけ変換不可にする
            if returned_source_kana != source_kana:
                print(
                    "\n【警告】Geminiが入力文字列を変更しました。"
                )
                print(f"index: {index}")
                print(f"入力: {source_kana}")
                print(f"回答: {item.source_kana}")
                print("この候補は変換不可として処理を続行します。")

                converted_sentences.append(
                    "変換不可"
                )
                model_valid_results.append(
                    False
                )
                reading_match_results.append(
                    False
                )

                continue

            if item.valid and item.converted != "変換不可":
                reading_match = readings_match(
                    source_kana,
                    item.converted,
                    sudachi_tokenizer,
                )
            else:
                reading_match = False

            # Geminiが有効と判定し、
            # さらに機械的な読み確認にも合格した場合だけ採用
            if item.valid and reading_match:
                final_converted = item.converted
            else:
                final_converted = "変換不可"

            converted_sentences.append(
                final_converted
            )
            model_valid_results.append(
                item.valid
            )
            reading_match_results.append(
                reading_match
            )

        converted_text = "\n".join(
            converted_sentences
        )

        converted_output_path.write_text(
            converted_text,
            encoding="utf-8",
        )

        # 第1～第3段階の対応関係をTSVに保存
        write_stage3_tsv(
            output_path=result_output_path,
            original_sentences=original_sentences,
            kana_sentences=kana_sentences_from_file,
            no_ma_sentences=no_ma_sentences,
            converted_sentences=converted_sentences,
            model_valid_results=model_valid_results,
            reading_match_results=reading_match_results,
        )

        # ==========================================
        # 第4段階：両文の読みを機械判定
        # ==========================================

        kinta_readings: list[str] = []
        kintama_readings: list[str] = []
        full_reading_match_results: list[bool] = []

        with measure_time("第4段階：読みの機械判定"):
            for original, converted in zip(
                original_sentences,
                converted_sentences,
            ):
                (
                    kinta_reading,
                    kintama_reading,
                    full_reading_match,
                ) = compare_full_dajare_readings(
                    original,
                    converted,
                    sudachi_tokenizer,
                )

                kinta_readings.append(
                    kinta_reading
                )
                kintama_readings.append(
                    kintama_reading
                )
                full_reading_match_results.append(
                    full_reading_match
                )


        # ==========================================
        # 第4A段階：Geminiによる成立性判定
        # ==========================================

        with measure_time("第4A段階：成立性判定"):
            validity_result = evaluate_validity_with_gemini(
                client,
                original_sentences,
                converted_sentences,
            )

        validity_by_index = validate_result_indexes(
            validity_result.items,
            len(original_sentences),
            "第4A段階",
        )


        # ==========================================
        # 面白さ採点の対象を選定
        # ==========================================

        humor_target_indexes: list[int] = []

        for index, converted in enumerate(
            converted_sentences,
            start=1,
        ):
            validity = validity_by_index[index]

            work_valid = (
                converted != "変換不可"
                and full_reading_match_results[index - 1]
                and validity.kinta_valid
                and validity.kintama_valid
            )

            if work_valid:
                humor_target_indexes.append(
                    index
                )


        # ==========================================
        # 第4B段階：成立した作品だけを面白さ採点
        # ==========================================

        if humor_target_indexes:
            with measure_time("第4B段階：面白さ採点"):
                humor_result = score_humor_with_gemini(
                    client,
                    original_sentences,
                    converted_sentences,
                    humor_target_indexes,
                )

            humor_by_index = validate_exact_result_indexes(
                humor_result.items,
                set(humor_target_indexes),
                "第4B段階",
            )

        else:
            humor_result = HumorResponse(
                items=[]
            )
            humor_by_index = {}

            print(
                "成立性判定に合格した候補がないため、"
                "面白さ採点を省略しました。"
            )

        write_stage4_tsv(
            output_path=evaluation_output_path,
            original_sentences=original_sentences,
            kana_sentences=kana_sentences_from_file,
            no_ma_sentences=no_ma_sentences,
            converted_sentences=converted_sentences,
            stage3_model_valid_results=model_valid_results,
            stage3_reading_match_results=reading_match_results,
            validity_items=validity_result.items,
            humor_items=humor_result.items,
            kinta_readings=kinta_readings,
            kintama_readings=kintama_readings,
            full_reading_match_results=full_reading_match_results,
        )
        # ==========================================
        # 第4段階の評価結果を累積履歴へ追記
        # ==========================================

        stage4_saved_at = datetime.now()

        stage4_run_id = stage4_saved_at.strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        stage4_executed_at = stage4_saved_at.isoformat(
            timespec="seconds"
        )

        stage4_history_count = append_stage4_history_tsv(
            latest_output_path=evaluation_output_path,
            history_output_path=evaluation_history_output_path,
            run_id=stage4_run_id,
            executed_at=stage4_executed_at,
            test_mode=TEST_MODE,
            model_name=MODEL_NAME,
        )

        print(
            f"第4段階の累積履歴追記件数: "
            f"{stage4_history_count}件"
        )

        print(
            f"第4段階の累積履歴: "
            f"{evaluation_history_output_path.resolve()}"
        )

        # ==========================================
        # 80点以上の作品だけを累積保存
        # ==========================================

        (
            qualified_high_score_count,
            added_high_score_count,
            duplicate_high_score_count,
        ) = append_high_score_works_tsv(
            latest_output_path=evaluation_output_path,
            high_score_output_path=high_score_output_path,
            run_id=stage4_run_id,
            executed_at=stage4_executed_at,
            model_name=MODEL_NAME,
            threshold=HIGH_SCORE_THRESHOLD,
        )

        print(
            f"{HIGH_SCORE_THRESHOLD}点以上の条件該当件数: "
            f"{qualified_high_score_count}件"
        )

        print(
            f"{HIGH_SCORE_THRESHOLD}点以上の新規追加件数: "
            f"{added_high_score_count}件"
        )

        print(
            f"既登録のため追加しなかった件数: "
            f"{duplicate_high_score_count}件"
        )

        print(
            f"高得点作品の累積ファイル: "
            f"{high_score_output_path.resolve()}"
        )

        # ==========================================
        # 実行結果を表示
        # ==========================================

        print("第1段階から第4段階まで完了しました。")

        print(f"\n候補文: {output_path}")
        print(f"ひらがな: {kana_output_path}")
        print(f"まを削除: {no_ma_output_path}")
        print(f"Gemini変換結果: {converted_output_path}")
        print(f"第3段階対応表: {result_output_path}")
        print(f"第4段階評価表: {evaluation_output_path}")

        print(
            f"第4段階累積評価表: "
            f"{evaluation_history_output_path}"
        )

        print(
            f"{HIGH_SCORE_THRESHOLD}点以上の作品集: "
            f"{high_score_output_path}"
        )

        print("\n【第1段階：生成した候補文】")
        print(candidate_text)

        print("\n【第2段階：ひらがな変換】")
        print(kana_text)

        print("\n【第3段階：先頭の『ま』を削除】")
        print(no_ma_text)

        print("\n【第3段階：Gemini変換結果】")
        print(converted_text)

        print("\n【第4段階：評価結果】")

        for index, (original, converted) in enumerate(
            zip(
                original_sentences,
                converted_sentences,
            ),
            start=1,
        ):
            validity = validity_by_index[index]

            full_reading_match = (
                full_reading_match_results[index - 1]
            )

            # 作品全体が成立しているか
            work_valid = (
                converted != "変換不可"
                and full_reading_match
                and validity.kinta_valid
                and validity.kintama_valid
            )

            if work_valid:
                humor = humor_by_index[index]

                total_score = (
                    humor.meaning_gap
                    + humor.imagery
                    + humor.punch
                    + humor.rhythm
                    + humor.surprise
                    + humor.originality
                )

                score_detail = (
                    f"意味の落差={humor.meaning_gap}/25、"
                    f"情景={humor.imagery}/25、"
                    f"オチ={humor.punch}/20、"
                    f"リズム={humor.rhythm}/15、"
                    f"意外性={humor.surprise}/10、"
                    f"独創性={humor.originality}/5"
                )

                humor_comment = humor.comment

            else:
                total_score = 0
                score_detail = "採点対象外"
                humor_comment = (
                    "成立性判定または読み一致判定で不合格"
                )

            print(
                f"{index}. "
                f"作品成立={work_valid} / "
                f"面白さ={total_score}点 / "
                f"読み完全一致={full_reading_match}"
            )

            print(
                f"   金太側成立={validity.kinta_valid}"
                f"（{validity.kinta_reason}）"
            )

            print(
                f"   金太、{original}"
            )

            print(
                f"   キンタマ側成立={validity.kintama_valid}"
                f"（{validity.kintama_reason}）"
            )

            if converted == "変換不可":
                print(
                    "   キンタマ側：変換不可"
                )
            else:
                print(
                    f"   キンタマ、{converted}"
                )

            print(
                f"   得点内訳：{score_detail}"
            )

            print(
                f"   講評：{humor_comment}"
            )

    finally:
        client.close()


if __name__ == "__main__":
    main()
