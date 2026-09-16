"""Case Agent final JSON output contract."""

OUTPUT_CONTRACT = {
    "type": "object",
    "required": [
        "display_mode",
        "content_blocks",
        "sources",
        "fallback_notice",
        "metadata",
    ],
    "additionalProperties": False,
    "properties": {
        "display_mode": {
            "type": "string",
            "enum": ["plain", "grounded", "unavailable", "error"],
        },
        "content_blocks": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "required": ["text", "source_refs"],
                "additionalProperties": False,
                "properties": {
                    "text": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1200,
                    },
                    "source_refs": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "sources": {
            "type": "array",
            "maxItems": 20,
            "items": {
                "type": "object",
                "required": [
                    "source_ref",
                    "source_type",
                    "title",
                    "version",
                    "detail",
                    "metadata",
                ],
                "additionalProperties": False,
                "properties": {
                    "source_ref": {"type": "string", "maxLength": 200},
                    "source_type": {"type": "string", "maxLength": 80},
                    "title": {"type": "string", "maxLength": 200},
                    "version": {"type": ["string", "null"], "maxLength": 80},
                    "detail": {
                        "type": "object",
                        "required": ["fields"],
                        "additionalProperties": False,
                        "properties": {
                            "fields": {
                                "type": "array",
                                "maxItems": 12,
                                "items": {
                                    "type": "object",
                                    "required": ["label", "value"],
                                    "additionalProperties": False,
                                    "properties": {
                                        "label": {"type": "string", "maxLength": 80},
                                        "value": {"type": "string", "maxLength": 500},
                                    },
                                },
                            }
                        },
                    },
                    "metadata": {"type": "object"},
                },
            },
        },
        "answer_markdown": {"type": ["string", "null"], "maxLength": 4000},
        "claims": {
            "type": "array",
            "maxItems": 30,
            "items": {
                "type": "object",
                "required": [
                    "claim_id",
                    "need_id",
                    "need_text",
                    "text",
                    "fact_refs",
                    "source_refs",
                    "citation_ids",
                    "support_status",
                ],
                "additionalProperties": False,
                "properties": {
                    "claim_id": {"type": "string", "maxLength": 80},
                    "need_id": {"type": "string", "maxLength": 80},
                    "need_text": {"type": "string", "maxLength": 240},
                    "text": {"type": "string", "maxLength": 600},
                    "fact_refs": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {"type": "string"},
                    },
                    "source_refs": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {"type": "string"},
                    },
                    "citation_ids": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {"type": "string"},
                    },
                    "support_status": {"type": "string", "maxLength": 40},
                },
            },
        },
        "citations": {
            "type": "array",
            "maxItems": 30,
            "items": {
                "type": "object",
                "required": [
                    "citation_id",
                    "label",
                    "claim_id",
                    "fact_refs",
                    "evidence_refs",
                    "source_refs",
                ],
                "additionalProperties": False,
                "properties": {
                    "citation_id": {"type": "string", "maxLength": 80},
                    "label": {"type": "integer", "minimum": 1, "maximum": 99},
                    "claim_id": {"type": ["string", "null"], "maxLength": 80},
                    "fact_refs": {
                        "type": "array",
                        "maxItems": 12,
                        "items": {"type": "string"},
                    },
                    "evidence_refs": {
                        "type": "array",
                        "maxItems": 12,
                        "items": {"type": "string"},
                    },
                    "source_refs": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "fallback_notice": {"type": "string", "maxLength": 500},
        "metadata": {
            "type": "object",
            "required": ["intent", "requires_citation", "capabilities_used"],
            "additionalProperties": True,
            "properties": {
                "intent": {"type": "string"},
                "requires_citation": {"type": "boolean"},
                "capabilities_used": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
    },
    "mode_rules": {
        "plain": "用于普通功能说明或会话解释。sources 必须为 []；所有 content_blocks.source_refs 必须为 []；fallback_notice 必须为空字符串。",
        "unavailable": "用于能力未接入、数据未生成或不可用。sources 必须为 []；所有 content_blocks.source_refs 必须为 []；fallback_notice 写简短原因；不得用通用知识补答。",
        "error": "用于错误或安全降级。sources 必须为 []；所有 content_blocks.source_refs 必须为 []；fallback_notice 写友好降级说明；不得暴露内部异常。",
        "grounded": "用于已读取当前案件来源的回答。sources 必须逐项复制 available_sources 中的完整对象；content_blocks.source_refs、claims.source_refs、citations.source_refs 只能引用 sources 中已有 source_ref；如使用 answer_markdown，每个事实句后用 [1] 这类角标绑定 citations。",
    },
    "content_block_count_policy": {
        "single_fact": "1 个 content_block。",
        "general_help": "1 到 2 个 content_blocks。",
        "case_explanation": "2 到 3 个 content_blocks。",
        "comprehensive_analysis": "最多 4 个 content_blocks。",
    },
    "forbidden_top_level_fields": [
        "brief_answer",
        "references",
        "summary",
        "suggestions",
    ],
}
