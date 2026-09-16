"""LLM-visible registry describing the Policy RAG metadata taxonomy."""

from __future__ import annotations

import json
from typing import Any


POLICY_FILTER_CATALOG: dict[str, Any] = {
    "jurisdiction": {
        "allowed": ["national", "beijing", "shanghai"],
        "guidance": (
            "明确城市优先；国家统一制度和国家级监管规则使用 national；"
            "跨地区问题可以多选；只有“本市”而缺少可信地区上下文时不要猜城市；"
            "问题明确写出北京或上海时，不要为了异地/跨省字样自动追加 national；"
            "当前语料中‘零星报销’是上海经办口径，即使问题未再次写明城市也使用 shanghai。"
        ),
    },
    "content_type": {
        "policy_text": "政策规则、适用范围、待遇条件、监管认定和解释性正文。",
        "table_row": "具体药品、项目、耗材、机构的编码、类别、比例、价格或名录行。",
        "boundary": (
            "content_type 只描述 chunk 结构；办事指南和问答使用 doc_type 表达，"
            "其正文 chunk 仍选择 policy_text。"
        ),
    },
    "policy_domain": {
        "benefit": {
            "meaning": "参保待遇、起付标准、支付比例、封顶线和待遇资格。",
            "boundary": "异地规则本身使用 remote_medical。",
        },
        "chronic_disease_long_prescription": {
            "meaning": "慢性病长期处方、续方和长期处方管理。",
            "boundary": "特殊病备案和病种范围分别使用对应 special_disease 领域。",
        },
        "designated_institution": {
            "meaning": "具体医院或药店的定点名录、名称、编码、地址和状态。",
            "boundary": "提到定点机构但询问费用审核或违规行为时使用 fund_supervision。",
        },
        "drug_catalog": {
            "meaning": "医保药品目录、甲乙类、目录编号、限定支付范围和具体药品的本地支付比例。",
            "boundary": (
                "药店实际销售价格参考使用 drug_product_price_reference；"
                "上海具体药品行的医保类别或本地支付比例仍只使用 drug_catalog + table_row，"
                "不要追加 shanghai_payment_scope。"
            ),
        },
        "drug_product_price_reference": {
            "meaning": "上海医保药店药品产品均价和价格区间参考。",
            "boundary": "不是报销政策依据，必须 can_cite_as_policy_basis=false。",
        },
        "emergency": {
            "meaning": "急诊抢救、急诊留观等独立政策规则。",
            "boundary": "问题仅在报销场景中提到急诊时，不要自动追加本领域。",
        },
        "fund_supervision": {
            "meaning": "基金使用监管、费用审核方式、重复申报、冒名就医和违规行为规则。",
            "boundary": "只查询具体机构名录时使用 designated_institution。",
        },
        "manual_reimbursement": {
            "meaning": "本地手工或零星报销的材料、条件、票据、办结时限和办理要求。",
            "boundary": (
                "跨省异地费用的手工报销使用 remote_medical_manual_reimbursement；"
                "当前语料的零星报销问题使用 shanghai + manual_reimbursement + policy_text，"
                "材料或时限措辞本身不触发 service_guide；"
                "北京审核员核验手工报销材料/结算凭证的清单类问题使用 table_row。"
            ),
        },
        "medical_service_price": {
            "meaning": "具体医疗服务项目编码、计价单位、收费标准和医保类别。",
            "boundary": "普通政策正文出现“项目”二字不能触发本领域。",
        },
        "remote_medical": {
            "meaning": "跨省异地就医备案、直接结算、就医地目录与参保地待遇分工。",
            "boundary": "异地费用线下手工报销可同时使用 remote_medical_manual_reimbursement。",
        },
        "remote_medical_manual_reimbursement": {
            "meaning": "跨省异地就医未直接结算后的手工报销规则。",
            "boundary": (
                "普通本地手工报销使用 manual_reimbursement；"
                "问题以审核员视角询问急诊留观、补办备案后的审核要点、不能直接结算原因或报销路径时，"
                "按当前语料使用 table_row；一般性‘如何报销’说明才使用 policy_text。"
            ),
        },
        "shanghai_payment_scope": {
            "meaning": "上海药品、诊疗项目和医用耗材的医保支付范围规则。",
            "boundary": (
                "北京具体项目价格使用 medical_service_price；"
                "查询上海某个具体药品目录行的类别或本地支付比例使用 drug_catalog，"
                "不能追加本领域。"
            ),
        },
        "special_disease_filing": {
            "meaning": "门诊特殊病备案资格、备案权限和备案办理规则。",
            "boundary": "外埠户籍本身不能触发；询问病种清单使用 special_disease_scope。",
        },
        "special_disease_scope": {
            "meaning": "门诊特殊疾病病种范围和具体疾病是否属于范围。",
            "boundary": (
                "备案流程使用 special_disease_filing；如果问题问的是某疾病的限定支付条件，"
                "即使句子提到使用某种药品，也不要追加 drug_catalog/table_row。"
            ),
        },
    },
}


POLICY_FILTER_FEW_SHOTS = [
    {
        "question": "医保经办机构对定点医药机构申报费用可以采取哪些审核方式？",
        "filters": {
            "jurisdiction": ["national"],
            "policy_domain": ["fund_supervision"],
            "content_type": ["policy_text"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "这两家北京药店是否都在医保定点零售药店名录中？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["designated_institution"],
            "content_type": ["table_row"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "项目编码W0311030010的计价单位、收费标准和医保类别是什么？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["medical_service_price"],
            "content_type": ["table_row"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "普通片剂具体包含哪些剂型？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["drug_catalog"],
            "content_type": ["policy_text"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "双氯芬酸钠滴眼剂的医保类别和目录编号是什么？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["drug_catalog"],
            "content_type": ["table_row"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "参保人零星报销门诊费用时要提供什么材料，法定办结时限多久？",
        "filters": {
            "jurisdiction": ["shanghai"],
            "policy_domain": ["manual_reimbursement"],
            "content_type": ["policy_text"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "上海医保药品目录中某个具体药品的医保类别和本地支付比例是多少？",
        "filters": {
            "jurisdiction": ["shanghai"],
            "policy_domain": ["drug_catalog"],
            "content_type": ["table_row"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "北京参保人在上海跨省直接结算时，支付范围与待遇政策如何分工？",
        "filters": {
            "jurisdiction": ["beijing", "shanghai", "national"],
            "policy_domain": ["remote_medical"],
            "content_type": ["policy_text"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "北京参保人员跨省异地就医直接结算时，医疗费用支付规则怎样，备案成功后可在哪些机构就医？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["remote_medical"],
            "content_type": ["policy_text"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "审核北京参保人员手工报销时，涉及外埠就医、定点医药机构记账结算和急诊就医，应核验哪些材料或结算凭证？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["manual_reimbursement"],
            "content_type": ["table_row"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "北京参保人员外地急诊留观自费后补办跨省备案，现申请医保手工报销，审核时要确认哪些政策要点？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["remote_medical_manual_reimbursement"],
            "content_type": ["table_row"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "门诊特殊疾病患者因类风湿关节炎使用新增报销范围药品，应满足哪些限定支付条件？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["special_disease_scope"],
            "content_type": ["policy_text"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "审核一笔城乡居民医保老年患者的慢性病长期处方费用，需要核验参保资格以及医疗机构的长期处方品规衔接要求？",
        "filters": {
            "policy_domain": ["benefit", "chronic_disease_long_prescription"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "审核本市社区卫生服务站的高血压长期处方费用，既要核验定点属性，也要确认糖尿病是否属于按人头付费慢性病门诊试点范围。",
        "filters": {
            "policy_domain": [
                "chronic_disease_long_prescription",
                "designated_institution",
            ],
            "content_type": ["policy_text", "table_row"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "北京参保人员异地急诊留观费用申请手工报销，同时核验医疗机构长处方政策造成的医事服务费损失补偿。",
        "filters": {
            "policy_domain": [
                "chronic_disease_long_prescription",
                "remote_medical_manual_reimbursement",
            ],
            "content_type": ["policy_text", "table_row"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "高血压患者就诊于按人头付费医联体，同时申请重性精神病门诊特殊疾病备案，请核验两项政策依据。",
        "filters": {
            "policy_domain": [
                "chronic_disease_long_prescription",
                "special_disease_scope",
            ],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "审核城乡居民医保门急诊费用，需要确认门急诊起付标准和支付比例，另核验一个医疗服务项目的医保类别。",
        "filters": {
            "policy_domain": ["benefit", "medical_service_price"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "北京参保人员在异地急诊留观后申请手工报销，同时确认上海医疗服务设施项目的医保基金支付范围。",
        "filters": {
            "jurisdiction": ["beijing", "shanghai"],
            "policy_domain": [
                "remote_medical_manual_reimbursement",
                "shanghai_payment_scope",
            ],
            "content_type": ["policy_text", "table_row"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "北京参保人员在异地跨省联网定点机构使用长期处方，审核直接结算时同时依据本市长期处方政策和跨省直接结算规定。",
        "filters": {
            "jurisdiction": ["beijing", "national"],
            "policy_domain": [
                "chronic_disease_long_prescription",
                "remote_medical",
            ],
            "content_type": ["policy_text"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "审核北京机构发生的长期处方费用，既核验定点目录信息，也要说明拒不配合基金监管调查的处理依据。",
        "filters": {
            "jurisdiction": ["beijing", "national"],
            "policy_domain": [
                "chronic_disease_long_prescription",
                "designated_institution",
                "fund_supervision",
            ],
            "content_type": ["policy_text", "table_row"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "外埠户籍配偶办理城乡居民医保需要哪些材料？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["benefit"],
            "content_type": ["policy_text"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "特殊病备案权限应如何管理？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["special_disease_filing"],
            "content_type": ["policy_text"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "哪些疾病属于门诊特殊疾病范围？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["special_disease_scope"],
            "content_type": ["policy_text"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "上海新版医保目录从何时开始执行？",
        "filters": {
            "jurisdiction": ["shanghai"],
            "policy_domain": ["shanghai_payment_scope"],
            "content_type": ["policy_text"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "跨省异地就医未直接结算后申请手工报销需要哪些材料？",
        "filters": {
            "jurisdiction": ["national", "beijing"],
            "policy_domain": ["remote_medical_manual_reimbursement"],
            "content_type": ["policy_text"],
            "can_cite_as_policy_basis": True,
        },
    },
    {
        "question": "上海医保药店中某药品上周的均价和价格区间是多少？",
        "filters": {
            "jurisdiction": ["shanghai"],
            "policy_domain": ["drug_product_price_reference"],
            "content_type": ["table_row"],
            "can_cite_as_policy_basis": False,
        },
    },
]


def policy_filter_prompt_contract() -> str:
    """Render the shared catalog and examples for the model prompt."""

    return json.dumps(
        {
            "filter_catalog": POLICY_FILTER_CATALOG,
            "examples": POLICY_FILTER_FEW_SHOTS,
            "rules": [
                "先确定 answer_mode，再拆 information_needs。",
                "information_needs 是用户真正需要回答的内容，不是固定槽位名。",
                "多问拆成多个 information_needs，保持每条足够具体。",
                "只选择回答问题确实需要的领域，不为提高召回追加相似领域。",
                "提到某类主体不等于查询该主体的名录。",
                "复合题的 filters 是各实际证据来源 metadata 的并集；若题目明确要求结合本市规则和国家/跨省统一规则，才同时选择对应地区。",
                "复合题可以省略无法从问题确定的地区或内容类型；不要为了完整而猜测或补齐字段。",
                "多领域数组由 Policy RAG MCP 自行分组，不要在这里生成子查询。",
                "输出必须使用注册表中的英文规范值。",
                "content_type 按检索语料的实际类型选择，不要根据问题措辞机械扩张。",
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
