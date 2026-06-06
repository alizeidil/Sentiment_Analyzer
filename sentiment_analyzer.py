"""
Türkçe duygu analizi (sentiment analysis) modülü.

Hugging Face üzerindeki savasy/bert-base-turkish-sentiment-cased modeli kullanılır.
Bu model, dbmdz/bert-base-turkish-cased (BERTurk) üzerine Türkçe metinler için
fine-tune edilmiştir ve yorumları olumlu (pozitif) veya olumsuz (negatif) olarak sınıflandırır.
"""

from dataclasses import dataclass

from transformers import pipeline

BASE_MODEL = "dbmdz/bert-base-turkish-cased"
MODEL_ID = "savasy/bert-base-turkish-sentiment-cased"

LABEL_TR = {
    "positive": "Olumlu (Pozitif)",
    "negative": "Olumsuz (Negatif)",
}


@dataclass
class SentimentResult:
    yorum: str
    etiket: str
    etiket_tr: str
    guven_skoru: float


class SentimentAnalyzer:
    """Hazır BERT tabanlı Türkçe duygu analizi sınıfı."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self._classifier = pipeline(
            "text-classification",
            model=model_id,
            tokenizer=model_id,
        )

    def tahmin_et(self, yorum: str) -> SentimentResult:
        """Tek bir yorumun duygusunu tahmin eder."""
        yorum = yorum.strip()
        if not yorum:
            raise ValueError("Yorum boş olamaz.")

        sonuc = self._classifier(yorum)[0]
        etiket = sonuc["label"].lower()

        return SentimentResult(
            yorum=yorum,
            etiket=etiket,
            etiket_tr=LABEL_TR.get(etiket, etiket),
            guven_skoru=round(float(sonuc["score"]), 4),
        )

    def tahmin_et_toplu(self, yorumlar: list[str]) -> list[SentimentResult]:
        """Birden fazla yorumu sırayla analiz eder."""
        return [self.tahmin_et(yorum) for yorum in yorumlar]
