#!/usr/bin/env python3
"""
Emotion and sentiment analytics utilities for Phase 15.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Tuple

from transformers import pipeline

logger = logging.getLogger("kalki.analytics.emotion")

QUESTION_WORDS = {"why", "how", "what", "where", "when", "who", "whom", "whose"}
CURIOSITY_TERMS = {"curious", "wonder", "explore", "discover", "learn", "investigate", "explain"}
STRESS_TERMS = {"anxious", "worried", "stressed", "overwhelmed", "panic", "pressure", "nervous", "afraid"}
POSITIVE_TERMS = {"great", "good", "love", "excellent", "happy", "joy", "fantastic", "excited", "grateful"}
NEGATIVE_TERMS = {"bad", "terrible", "sad", "angry", "hate", "awful", "upset", "frustrated", "worried", "tired"}
ANGER_TERMS = {"angry", "furious", "rage", "irritated", "mad"}
FEAR_TERMS = {"afraid", "scared", "terrified", "nervous", "anxious"}
SADNESS_TERMS = {"sad", "unhappy", "depressed", "down", "lonely"}
JOY_TERMS = {"happy", "joyful", "pleased", "excited", "glad"}


class SentimentAnalyzer:
    """Real sentiment and emotion analyzer built on open-source HF models."""

    def __init__(self) -> None:
        self._sentiment_pipeline = None
        self._emotion_pipeline = None
        self._load_lock = asyncio.Lock()
        self._use_fallback = False

    async def analyze(self, text: str) -> Dict[str, Any]:
        if not text.strip():
            raise ValueError("Text must be non-empty for sentiment analysis")

        await self._ensure_pipelines_loaded()

        loop = asyncio.get_running_loop()
        sentiment_result = await loop.run_in_executor(None, self._run_sentiment, text)
        emotion_scores = await loop.run_in_executor(None, self._run_emotion, text)

        sentiment_score = self._score_from_sentiment(sentiment_result)
        primary_emotion, intensity = self._select_primary_emotion(emotion_scores)
        stress_level = self._estimate_stress(primary_emotion, emotion_scores, text)
        curiosity_level = self._estimate_curiosity(text)
        empathy_alignment = self._estimate_empathy(sentiment_score, stress_level, primary_emotion)

        return {
            "sentiment_label": sentiment_result["label"],
            "sentiment_score": sentiment_score,
            "emotion_scores": emotion_scores,
            "primary_emotion": primary_emotion,
            "emotion_intensity": intensity,
            "stress_level": stress_level,
            "curiosity_level": curiosity_level,
            "empathy_alignment": empathy_alignment,
        }

    def _run_sentiment(self, text: str) -> Dict[str, Any]:
        if self._use_fallback:
            return self._fallback_sentiment(text)
        result = self._sentiment_pipeline(text, truncation=True)[0]
        return {"label": result["label"], "score": float(result["score"])}

    def _run_emotion(self, text: str) -> List[Dict[str, Any]]:
        if self._use_fallback:
            return self._fallback_emotion(text)
        outputs = self._emotion_pipeline(text, truncation=True, return_all_scores=True)[0]
        return [{"label": item["label"], "score": float(item["score"])} for item in outputs]

    def _score_from_sentiment(self, sentiment_result: Dict[str, Any]) -> float:
        label = sentiment_result["label"].upper()
        score = sentiment_result["score"]
        return score if "POS" in label else -score

    def _select_primary_emotion(self, emotion_scores: List[Dict[str, Any]]) -> Tuple[str, float]:
        primary = max(emotion_scores, key=lambda x: x["score"])
        return primary["label"], float(primary["score"])

    def _estimate_stress(self, primary_emotion: str, emotion_scores: List[Dict[str, Any]], text: str) -> float:
        stress_emotions = {"anger", "fear", "sadness", "anxiety"}
        stress_score = 0.0
        for item in emotion_scores:
            if item["label"].lower() in stress_emotions:
                stress_score += item["score"]
        stress_score = min(stress_score, 1.0)

        text_lower = text.lower()
        term_hits = sum(text_lower.count(term) for term in STRESS_TERMS)
        stress_score += min(0.3, term_hits * 0.05)

        if primary_emotion.lower() in stress_emotions:
            stress_score = min(1.0, stress_score + 0.1)
        return float(min(1.0, stress_score))

    def _estimate_curiosity(self, text: str) -> float:
        text_lower = text.lower()
        tokens = re.findall(r"\w+", text_lower)
        if not tokens:
            return 0.0

        question_marks = text.count("?")
        question_word_hits = sum(1 for token in tokens if token in QUESTION_WORDS)
        curiosity_hits = sum(1 for token in tokens if token in CURIOSITY_TERMS)

        curiosity_score = 0.0
        curiosity_score += min(0.4, question_marks * 0.1)
        curiosity_score += min(0.4, question_word_hits * 0.05)
        curiosity_score += min(0.4, curiosity_hits * 0.08)
        curiosity_score = min(1.0, curiosity_score)

        return float(curiosity_score)

    def _estimate_empathy(self, sentiment_score: float, stress_level: float, primary_emotion: str) -> float:
        base = (sentiment_score + 1) / 2  # map [-1,1] -> [0,1]
        stress_penalty = stress_level * 0.4
        negative_bias = 0.2 if primary_emotion.lower() in {"anger", "disgust"} else 0.0
        empathy = max(0.0, min(1.0, base - stress_penalty - negative_bias))
        return float(empathy)

    async def _ensure_pipelines_loaded(self) -> None:
        if self._sentiment_pipeline and self._emotion_pipeline:
            return

        async with self._load_lock:
            if self._sentiment_pipeline and self._emotion_pipeline:
                return
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._load_pipelines)
            logger.info("Sentiment and emotion pipelines loaded for Emotion Analyzer")

    def _load_pipelines(self) -> None:
        try:
            if self._sentiment_pipeline is None:
                self._sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                )
            if self._emotion_pipeline is None:
                self._emotion_pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True,
                )
            self._use_fallback = False
        except Exception as exc:
            logger.warning("Falling back to heuristic emotion analysis: %s", exc)
            self._setup_rule_based_fallback()

    def _setup_rule_based_fallback(self) -> None:
        self._use_fallback = True
        self._sentiment_pipeline = None
        self._emotion_pipeline = None

    def _fallback_sentiment(self, text: str) -> Dict[str, Any]:
        tokens = re.findall(r"\w+", text.lower())
        pos_hits = sum(tokens.count(term) for term in POSITIVE_TERMS)
        neg_hits = sum(tokens.count(term) for term in NEGATIVE_TERMS)
        total = pos_hits + neg_hits
        if total == 0:
            score = 0.0
        else:
            score = (pos_hits - neg_hits) / total
        label = "POSITIVE" if score >= 0 else "NEGATIVE"
        return {"label": label, "score": abs(score) if total else 0.0}

    def _fallback_emotion(self, text: str) -> List[Dict[str, Any]]:
        tokens = re.findall(r"\w+", text.lower())
        counts = {
            "joy": sum(tokens.count(term) for term in JOY_TERMS),
            "anger": sum(tokens.count(term) for term in ANGER_TERMS),
            "fear": sum(tokens.count(term) for term in FEAR_TERMS),
            "sadness": sum(tokens.count(term) for term in SADNESS_TERMS),
        }
        total = sum(counts.values())
        if total == 0:
            counts = {key: 1 for key in ["joy", "anger", "fear", "sadness"]}
            total = sum(counts.values())
        scores = []
        for label in ["joy", "anger", "fear", "sadness"]:
            scores.append({"label": label, "score": counts[label] / total})
        neutral_score = max(0.0, 1.0 - sum(item["score"] for item in scores))
        scores.append({"label": "neutral", "score": neutral_score})
        return scores

    async def cleanup(self) -> None:
        self._sentiment_pipeline = None
        self._emotion_pipeline = None


__all__ = ["SentimentAnalyzer"]
