"""
LLM Service Layer for PsychTrend
Provides high-level functions for LLM-enhanced features
"""
import json
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime

from ollama_client import get_ollama_client, check_ollama_health
from llm_prompts import (
    SYSTEM_PROMPT_BASE,
    INPUT_NORMALIZATION_SYSTEM, INPUT_NORMALIZATION_TEMPLATE,
    QUESTION_ENHANCEMENT_SYSTEM, QUESTION_ENHANCEMENT_TEMPLATE,
    INSIGHT_EXPLANATION_SYSTEM, INSIGHT_EXPLANATION_TEMPLATE,
    REPORT_GENERATION_SYSTEM, REPORT_EXECUTIVE_SUMMARY_TEMPLATE, REPORT_FULL_TEMPLATE,
    STRENGTH_HUMANIZATION_TEMPLATE, GROWTH_HUMANIZATION_TEMPLATE,
    validate_output, sanitize_output
)
from ml_engine.sentiment_context import get_description_tone_guidance


class LLMService:
    """High-level LLM service for PsychTrend features"""
    
    def __init__(self):
        self.client = get_ollama_client()
        self._is_available: Optional[bool] = None
        self._last_health_check: Optional[datetime] = None
    
    async def is_available(self, force_check: bool = False) -> bool:
        """Check if LLM service is available"""
        # Cache health check for 60 seconds
        if not force_check and self._is_available is not None:
            if self._last_health_check:
                elapsed = (datetime.now() - self._last_health_check).seconds
                if elapsed < 60:
                    return self._is_available
        
        health = await check_ollama_health()
        self._is_available = health.get("status") == "healthy" and health.get("model_available", False)
        self._last_health_check = datetime.now()
        return self._is_available
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status"""
        return await check_ollama_health()
    
    # =========================================================================
    # INPUT NORMALIZATION
    # =========================================================================
    
    async def normalize_input(
        self,
        user_input: str,
        category: str
    ) -> Dict[str, Any]:
        """
        Normalize weak or unclear user inputs.
        
        Args:
            user_input: Raw user input text
            category: Current conversation category
            
        Returns:
            Dict with 'normalized', 'quality', 'original', and 'used_llm'
        """
        # Quick check for obviously good inputs (don't waste LLM call)
        if len(user_input.split()) >= 5 and len(user_input) >= 30:
            return {
                "original": user_input,
                "normalized": user_input,
                "quality": "high",
                "used_llm": False
            }
        
        # Check if LLM is available
        if not await self.is_available():
            return {
                "original": user_input,
                "normalized": user_input,
                "quality": self._estimate_quality(user_input),
                "used_llm": False,
                "fallback_reason": "LLM not available"
            }
        
        # Use LLM for normalization
        prompt = INPUT_NORMALIZATION_TEMPLATE.format(
            user_input=user_input,
            category=category
        )
        
        result = await self.client.generate(
            prompt=prompt,
            system_prompt=INPUT_NORMALIZATION_SYSTEM,
            temperature=0.1,  # Very low for consistency
            json_mode=True
        )
        
        if result.get("success"):
            try:
                response_text = result.get("response", "{}")
                parsed = json.loads(response_text)
                
                normalized = parsed.get("normalized")
                quality = parsed.get("quality", "medium")
                
                return {
                    "original": user_input,
                    "normalized": normalized if normalized else user_input,
                    "quality": quality,
                    "used_llm": True
                }
            except json.JSONDecodeError:
                pass
        
        # Fallback
        return {
            "original": user_input,
            "normalized": user_input,
            "quality": self._estimate_quality(user_input),
            "used_llm": False,
            "fallback_reason": "LLM parsing failed"
        }
    
    def _estimate_quality(self, text: str) -> str:
        """Estimate input quality without LLM"""
        if not text or len(text.strip()) < 3:
            return "low"
        
        word_count = len(text.split())
        if word_count < 3:
            return "low"
        elif word_count < 8:
            return "medium"
        return "high"
    
    # =========================================================================
    # QUESTION ENHANCEMENT
    # =========================================================================
    
    async def enhance_question(
        self,
        user_response: str,
        category: str,
        previous_question: str
    ) -> Optional[str]:
        """
        Generate an enhanced follow-up question.
        
        Returns:
            Enhanced question string, or None if LLM not available
        """
        if not await self.is_available():
            return None
        
        prompt = QUESTION_ENHANCEMENT_TEMPLATE.format(
            user_response=user_response,
            category=category,
            previous_question=previous_question
        )
        
        result = await self.client.generate(
            prompt=prompt,
            system_prompt=QUESTION_ENHANCEMENT_SYSTEM,
            temperature=0.3,
            max_tokens=150
        )
        
        if result.get("success"):
            question = result.get("response", "").strip()
            
            # Validate output
            is_valid, issues = validate_output(question)
            if not is_valid:
                question = sanitize_output(question)
            
            # Basic validation - should be a question
            if question and len(question) > 10:
                # Ensure it ends with ?
                if not question.endswith("?"):
                    question += "?"
                return question
        
        return None
    
    # =========================================================================
    # INSIGHT EXPLANATION
    # =========================================================================
    
    async def explain_insight(
        self,
        trend_name: str,
        score: float,
        direction: str,
        description: str
    ) -> str:
        """
        Generate human-readable explanation for a trend.
        
        Returns:
            Explanation string (falls back to template if LLM unavailable)
        """
        fallback = f"Your {trend_name.lower()} score is {score:.2f}, showing a {direction} trend. {description}"
        
        if not await self.is_available():
            return fallback
        
        prompt = INSIGHT_EXPLANATION_TEMPLATE.format(
            trend_name=trend_name,
            score=f"{score:.2f}",
            direction=direction,
            description=description
        )
        
        result = await self.client.generate(
            prompt=prompt,
            system_prompt=INSIGHT_EXPLANATION_SYSTEM,
            temperature=0.2,
            max_tokens=200
        )
        
        if result.get("success"):
            explanation = result.get("response", "").strip()
            
            # Validate
            is_valid, _ = validate_output(explanation)
            if not is_valid:
                explanation = sanitize_output(explanation)
            
            if explanation and len(explanation) > 20:
                return explanation
        
        return fallback
    
    # =========================================================================
    # REPORT GENERATION
    # =========================================================================
    
    async def generate_executive_summary(
        self,
        user_name: str,
        response_count: int,
        overall_sentiment: float,
        primary_archetype: Dict[str, Any],
        trends: Dict[str, Any]
    ) -> str:
        """Generate executive summary for report"""
        
        archetype_name = primary_archetype.get("cluster_name", "balanced").title()
        archetype_desc = primary_archetype.get("description", "Shows balanced behavioral patterns")
        
        fallback = (
            f"Based on {response_count} responses from {user_name}, the analysis reveals "
            f"a primarily '{archetype_name}' behavioral profile. {archetype_desc}. "
            f"Overall sentiment tendency is {overall_sentiment:.2f} "
            f"({'positive' if overall_sentiment > 0 else 'negative' if overall_sentiment < 0 else 'neutral'}). "
            f"Note: This is not a medical or psychological diagnosis."
        )
        
        if not await self.is_available():
            return fallback
        
        # Extract trend data
        motivation = trends.get("motivation", {})
        consistency = trends.get("consistency", {})
        growth = trends.get("growth", {})
        
        prompt = REPORT_EXECUTIVE_SUMMARY_TEMPLATE.format(
            user_name=user_name,
            response_count=response_count,
            overall_sentiment=f"{overall_sentiment:.2f}",
            primary_archetype=archetype_name,
            archetype_description=archetype_desc,
            motivation_score=f"{motivation.get('score', 0.5):.2f}",
            motivation_direction=motivation.get("trend_direction", "stable"),
            consistency_score=f"{consistency.get('score', 0.5):.2f}",
            consistency_direction=consistency.get("trend_direction", "stable"),
            growth_score=f"{growth.get('score', 0.5):.2f}",
            growth_direction=growth.get("trend_direction", "stable")
        )
        
        result = await self.client.generate(
            prompt=prompt,
            system_prompt=REPORT_GENERATION_SYSTEM,
            temperature=0.2,
            max_tokens=400
        )
        
        if result.get("success"):
            summary = result.get("response", "").strip()
            
            # Validate and sanitize
            is_valid, _ = validate_output(summary)
            if not is_valid:
                summary = sanitize_output(summary)
            
            if summary and len(summary) > 50:
                # Ensure disclaimer is present
                if "not a medical" not in summary.lower() and "not a psychological" not in summary.lower():
                    summary += "\n\nNote: This is not a medical or psychological diagnosis."
                return summary
        
        return fallback
    
    async def generate_full_report(
        self,
        user_name: str,
        response_count: int,
        trends: Dict[str, Any],
        clusters: Dict[str, Any],
        predictions: List[Dict[str, Any]],
        strengths: List[str],
        growth_areas: List[str]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive LLM-enhanced report.
        Includes sentiment context for stricter scoring alignment.
        
        Returns:
            Dict with report sections
        """
        # Get primary archetype
        archetypes = clusters.get("archetypes", [])
        primary = archetypes[0] if archetypes else {"cluster_name": "balanced", "description": "Balanced profile"}
        secondary = archetypes[1:3] if len(archetypes) > 1 else []
        
        # Extract sentiment context from trends or clusters
        sentiment_context = trends.get("sentiment_context", clusters.get("sentiment_context", {}))
        
        # Get attention areas and tone guidance
        attention_areas = sentiment_context.get("attention_areas", [])
        attention_areas_str = "\n".join(f"- {area}" for area in attention_areas) if attention_areas else "None identified"
        tone_guidance = get_description_tone_guidance(sentiment_context)
        
        # Prepare fallback response with attention areas
        fallback = {
            "executive_summary": await self.generate_executive_summary(
                user_name, response_count, 0.0, primary, trends
            ),
            "behavioral_profile": primary,
            "trend_explanations": {},
            "strengths": strengths,
            "growth_opportunities": growth_areas,
            "attention_areas": attention_areas,
            "llm_enhanced": False
        }
        
        if not await self.is_available():
            return fallback
        
        # Generate trend explanations
        trend_explanations = {}
        for trend_key, trend_data in trends.items():
            if isinstance(trend_data, dict) and "score" in trend_data:
                explanation = await self.explain_insight(
                    trend_name=trend_data.get("name", trend_key.replace("_", " ").title()),
                    score=trend_data.get("score", 0.5),
                    direction=trend_data.get("trend_direction", "stable"),
                    description=trend_data.get("description", "")
                )
                trend_explanations[trend_key] = explanation
        
        # Generate full report with attention areas and tone guidance
        prompt = REPORT_FULL_TEMPLATE.format(
            user_name=user_name,
            response_count=response_count,
            trends_json=json.dumps({k: v for k, v in trends.items() if k != 'sentiment_context'}, indent=2),
            primary_archetype=primary.get("cluster_name", "balanced").title(),
            archetype_description=primary.get("description", ""),
            secondary_archetypes=", ".join([a.get("cluster_name", "").title() for a in secondary]) or "None",
            predictions_json=json.dumps(predictions, indent=2),
            strengths_list="\n".join(f"- {s}" for s in strengths),
            growth_areas_list="\n".join(f"- {g}" for g in growth_areas),
            attention_areas=attention_areas_str,
            tone_guidance=tone_guidance
        )
        
        result = await self.client.generate(
            prompt=prompt,
            system_prompt=REPORT_GENERATION_SYSTEM,
            temperature=0.25,
            max_tokens=1800  # Increased for additional section
        )
        
        if result.get("success"):
            report_text = result.get("response", "").strip()
            
            # Validate and sanitize
            is_valid, _ = validate_output(report_text)
            if not is_valid:
                report_text = sanitize_output(report_text)
            
            if report_text and len(report_text) > 100:
                return {
                    "executive_summary": await self.generate_executive_summary(
                        user_name, response_count,
                        trends.get("motivation", {}).get("score", 0.5) - 0.5,
                        primary, trends
                    ),
                    "full_report_markdown": report_text,
                    "behavioral_profile": primary,
                    "trend_explanations": trend_explanations,
                    "strengths": strengths,
                    "growth_opportunities": growth_areas,
                    "attention_areas": attention_areas,
                    "llm_enhanced": True
                }
        
        # Return fallback with trend explanations we managed to generate
        fallback["trend_explanations"] = trend_explanations
        return fallback
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    async def humanize_strength(self, strength: str, user_name: str) -> str:
        """Make a strength statement more personal and encouraging"""
        if not await self.is_available():
            return strength
        
        prompt = STRENGTH_HUMANIZATION_TEMPLATE.format(
            strength=strength,
            user_name=user_name
        )
        
        result = await self.client.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT_BASE,
            temperature=0.3,
            max_tokens=100
        )
        
        if result.get("success"):
            humanized = result.get("response", "").strip()
            if humanized and len(humanized) > 10:
                is_valid, _ = validate_output(humanized)
                return humanized if is_valid else sanitize_output(humanized)
        
        return strength
    
    async def humanize_growth_area(self, growth_area: str, user_name: str) -> str:
        """Make a growth area statement more supportive"""
        if not await self.is_available():
            return growth_area
        
        prompt = GROWTH_HUMANIZATION_TEMPLATE.format(
            growth_area=growth_area,
            user_name=user_name
        )
        
        result = await self.client.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT_BASE,
            temperature=0.3,
            max_tokens=100
        )
        
        if result.get("success"):
            humanized = result.get("response", "").strip()
            if humanized and len(humanized) > 10:
                is_valid, _ = validate_output(humanized)
                return humanized if is_valid else sanitize_output(humanized)
        
        return growth_area


# Global service instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create global LLM service"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
