"""How many languages does a model actually have? Measured, per language.

A model card lists languages; this asks the model. The same passage, written in
each language by a human rather than machine-translated, scored with
`mynah-slm ppl`.

**The metric is BITS PER BYTE, not perplexity.** Perplexity is per TOKEN, and
two models with different vocabularies do not agree on what a token is: a
tokenizer that splits Japanese into many small pieces gets an easier job per
piece and a flattering perplexity for the same text. Bits per byte asks the
same question of the same bytes, so it compares across models.

**Read the table ACROSS a row, never down a column.** Bits per byte is
comparable between MODELS on the same text and not between LANGUAGES: Russian
and Chinese spend two or three UTF-8 bytes per character, so the denominator
grows and their bits/byte looks low for reasons that have nothing to do with
how well the model knows them. Within one row the bytes are identical and the
comparison is exact.

Lower is better.

    uv run python -m eval.lang_ppl ../models-local/granite-4.0-350m-Q8_0.gguf \\
                                   ../models-local/Qwen3-0.6B-Q4_K_M.gguf
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

# The same passage in each language. Kept short enough to run in a minute and
# long enough that a paragraph of context exists — a single sentence measures
# the tokenizer more than the model.
#
# The first twelve are the languages IBM lists for Granite 4.0. The last four
# are NOT on that list and are on Qwen3's, which is the point: a claim is only
# meaningful next to a language the model does not claim.
PASSAGES = {
    "en": "The ledger for 1887 lists twelve names, four of them women, and a column of figures nobody has explained since. Barges came up the river even in winter, loaded with coal for the furnaces that never went out. The factory ran day and night, and the town grew around it without anyone deciding that it should.",
    "de": "Das Kassenbuch von 1887 nennt zwölf Namen, vier davon Frauen, und eine Spalte mit Zahlen, die seither niemand erklärt hat. Die Kähne fuhren auch im Winter den Fluss hinauf, beladen mit Kohle für die Öfen, die niemals ausgingen. Die Fabrik lief Tag und Nacht, und die Stadt wuchs um sie herum, ohne dass jemand es beschlossen hätte.",
    "es": "El libro de cuentas de 1887 enumera doce nombres, cuatro de ellos mujeres, y una columna de cifras que nadie ha explicado desde entonces. Las barcazas remontaban el río incluso en invierno, cargadas de carbón para los hornos que nunca se apagaban. La fábrica funcionaba día y noche, y la ciudad creció a su alrededor sin que nadie lo decidiera.",
    "fr": "Le registre de 1887 énumère douze noms, dont quatre femmes, et une colonne de chiffres que personne n'a expliquée depuis. Les péniches remontaient le fleuve même en hiver, chargées de charbon pour les fours qui ne s'éteignaient jamais. L'usine tournait jour et nuit, et la ville a grandi autour d'elle sans que personne ne l'ait décidé.",
    "ja": "一八八七年の帳簿には十二の名前が並んでいる。そのうち四人は女性で、以来だれも説明していない数字の欄がある。艀は冬でも川をさかのぼり、決して消えない炉のための石炭を積んでいた。工場は昼も夜も動きつづけ、町はだれが決めたわけでもなくその周りに広がっていった。",
    "pt": "O livro de contas de 1887 lista doze nomes, quatro deles mulheres, e uma coluna de números que ninguém explicou desde então. As barcaças subiam o rio mesmo no inverno, carregadas de carvão para os fornos que nunca se apagavam. A fábrica funcionava dia e noite, e a cidade cresceu à sua volta sem que ninguém o tivesse decidido.",
    "ar": "يسرد دفتر الحسابات لعام ١٨٨٧ اثني عشر اسماً، أربعة منها لنساء، وعموداً من الأرقام لم يفسره أحد منذ ذلك الحين. كانت الصنادل تصعد النهر حتى في الشتاء، محملة بالفحم للأفران التي لم تنطفئ قط. كان المصنع يعمل ليلاً ونهاراً، ونمت المدينة حوله دون أن يقرر ذلك أحد.",
    "cs": "Účetní kniha z roku 1887 uvádí dvanáct jmen, čtyři z nich ženská, a sloupec čísel, který od té doby nikdo nevysvětlil. Čluny pluly proti proudu řeky i v zimě, naložené uhlím pro pece, které nikdy nevyhasly. Továrna běžela ve dne v noci a město kolem ní rostlo, aniž by to někdo rozhodl.",
    "it": "Il registro del 1887 elenca dodici nomi, quattro dei quali di donne, e una colonna di cifre che da allora nessuno ha più spiegato. Le chiatte risalivano il fiume anche d'inverno, cariche di carbone per le fornaci che non si spegnevano mai. La fabbrica andava giorno e notte, e la città le crebbe intorno senza che nessuno lo avesse deciso.",
    "ko": "1887년 장부에는 열두 개의 이름이 적혀 있고 그중 넷은 여자였으며, 그 뒤로 아무도 설명하지 못한 숫자 한 칸이 남아 있다. 거룻배는 겨울에도 강을 거슬러 올라왔고, 결코 꺼지지 않는 화로에 쓸 석탄을 싣고 있었다. 공장은 밤낮으로 돌아갔고, 도시는 누가 정한 것도 아닌데 그 둘레로 자라났다.",
    "nl": "Het kasboek van 1887 noemt twaalf namen, vier daarvan vrouwen, en een kolom cijfers die sindsdien niemand heeft verklaard. De schuiten voeren ook in de winter de rivier op, geladen met kolen voor de ovens die nooit uitgingen. De fabriek draaide dag en nacht, en de stad groeide eromheen zonder dat iemand daartoe besloot.",
    "zh": "一八八七年的账簿上列着十二个名字，其中四个是女人，还有一栏数字至今无人解释。驳船即使在冬天也逆流而上，装着送往那些从不熄灭的炉子的煤。工厂日夜运转，城镇就在它周围生长起来，并没有谁决定过。",
    # Not on Granite's list — the control.
    "ru": "В расчётной книге за 1887 год перечислены двенадцать имён, четыре из них женские, и столбец цифр, который с тех пор никто не объяснил. Баржи поднимались по реке даже зимой, гружённые углём для печей, которые никогда не гасли. Завод работал днём и ночью, и город рос вокруг него, хотя никто этого не решал.",
    "pl": "Księga rachunkowa z 1887 roku wymienia dwanaście nazwisk, cztery z nich kobiece, oraz kolumnę liczb, której od tamtej pory nikt nie wyjaśnił. Barki płynęły w górę rzeki nawet zimą, załadowane węglem do pieców, które nigdy nie gasły. Fabryka pracowała dzień i noc, a miasto rosło wokół niej, choć nikt tak nie postanowił.",
    "tr": "1887 tarihli defter on iki isim sıralar, bunlardan dördü kadındır, ve o günden beri kimsenin açıklamadığı bir rakam sütunu vardır. Mavnalar kışın bile nehri çıkardı, hiç sönmeyen ocaklar için kömür yüklüydüler. Fabrika gece gündüz çalıştı ve kasaba, kimse öyle karar vermeden onun çevresinde büyüdü.",
    "el": "Το κατάστιχο του 1887 απαριθμεί δώδεκα ονόματα, τα τέσσερα γυναικεία, και μια στήλη αριθμών που κανείς δεν εξήγησε έκτοτε. Οι φορτηγίδες ανέβαιναν το ποτάμι ακόμη και τον χειμώνα, φορτωμένες κάρβουνο για τα καμίνια που δεν έσβηναν ποτέ. Το εργοστάσιο δούλευε μέρα νύχτα, και η πόλη μεγάλωσε γύρω του χωρίς να το αποφασίσει κανείς.",
}

# What each model's card claims, so the measurement can be read against it.
CLAIMED = {
    "granite": set("en de es fr ja pt ar cs it ko nl zh".split()),
}

BITS = re.compile(r"bits/byte\s+([0-9.]+)")


def score(binary: str, model: str, text: str) -> float | None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        f.write(text)
        path = f.name
    try:
        p = subprocess.run([binary, "ppl", "-m", model, "-f", path, "--kv", "f32"],
                           capture_output=True, text=True, cwd=ROOT)
        m = BITS.search(p.stdout)
        return float(m.group(1)) if m else None
    finally:
        os.unlink(path)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    models = [os.path.abspath(m) for m in sys.argv[1:]]
    binary = os.path.join(ROOT, "mynah-slm")

    names = [os.path.basename(m).replace(".gguf", "") for m in models]
    print(f"{'lang':6s} " + " ".join(f"{n[:22]:>22s}" for n in names) + "   claimed")
    print("-" * (7 + 23 * len(names) + 12))

    totals = [0.0] * len(models)
    for lang, text in PASSAGES.items():
        row = []
        for i, m in enumerate(models):
            b = score(binary, m, text)
            row.append(b)
            if b is not None:
                totals[i] += b
        claim = "".join("G" if lang in CLAIMED["granite"] else "-")
        print(f"{lang:6s} " + " ".join(f"{b:22.4f}" if b else f"{'?':>22s}" for b in row)
              + f"   {claim}")

    print("-" * (7 + 23 * len(names) + 12))
    n = len(PASSAGES)
    print(f"{'mean':6s} " + " ".join(f"{t / n:22.4f}" for t in totals))
    print("\nbits per byte, lower is better. G = on Granite's claimed list.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
