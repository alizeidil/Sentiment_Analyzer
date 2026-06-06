"""
Türkçe Duygu Analizi Ödevi
--------------------------
Hugging Face modeli: savasy/bert-base-turkish-sentiment-cased
Temel model (referans): dbmdz/bert-base-turkish-cased

Kullanım:
    python main.py              # örnek yorumları çalıştırır
    python main.py --interaktif   # kullanıcıdan yorum alır
"""

import argparse
import sys

from sentiment_analyzer import BASE_MODEL, MODEL_ID, SentimentAnalyzer


ORNEK_YORUMLAR = [
    "Bu ürün harika, kesinlikle tavsiye ederim!",
    "Kargo çok geç geldi ve ürün beklentimi karşılamadı.",
    "Film mükemmeldi, oyunculuklar çok iyiydi.",
    "Müşteri hizmetleri berbat, bir daha alışveriş yapmam.",
    "Fiyatına göre idare eder, ne çok iyi ne çok kötü.",
]


def sonuc_yazdir(sonuc) -> None:
    print("-" * 50)
    print(f"Yorum      : {sonuc.yorum}")
    print(f"Tahmin     : {sonuc.etiket_tr}")
    print(f"Güven skoru: %{sonuc.guven_skoru * 100:.2f}")


def ornekleri_calistir(analyzer: SentimentAnalyzer) -> None:
    print("=" * 50)
    print("TÜRKÇE DUYGU ANALİZİ")
    print("=" * 50)
    print(f"Temel model : {BASE_MODEL}")
    print(f"Kullanılan  : {MODEL_ID}")
    print()

    for sonuc in analyzer.tahmin_et_toplu(ORNEK_YORUMLAR):
        sonuc_yazdir(sonuc)

    print("-" * 50)


def interaktif_mod(analyzer: SentimentAnalyzer) -> None:
    print("=" * 50)
    print("İNTERAKTİF DUYGU ANALİZİ")
    print("Çıkmak için 'q' yazın.")
    print("=" * 50)

    while True:
        try:
            yorum = input("\nYorumunuzu girin: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nProgram sonlandırıldı.")
            break

        if yorum.lower() in {"q", "quit", "çık", "cik"}:
            print("Program sonlandırıldı.")
            break

        if not yorum:
            print("Lütfen geçerli bir yorum girin.")
            continue

        try:
            sonuc_yazdir(analyzer.tahmin_et(yorum))
        except ValueError as hata:
            print(f"Hata: {hata}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Türkçe duygu analizi uygulaması")
    parser.add_argument(
        "--interaktif",
        action="store_true",
        help="Kullanıcıdan yorum alarak çalıştır",
    )
    args = parser.parse_args()

    print("Model yükleniyor, lütfen bekleyin...")
    try:
        analyzer = SentimentAnalyzer()
    except Exception as hata:
        print(f"Model yüklenemedi: {hata}", file=sys.stderr)
        sys.exit(1)

    if args.interaktif:
        interaktif_mod(analyzer)
    else:
        ornekleri_calistir(analyzer)


if __name__ == "__main__":
    main()
