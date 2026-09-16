"""Answer requirement registry for Policy Expert generation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.backend.application.agent.expert_agent.service.policy_answer_contracts import (
    AnswerRequirement,
)
from src.backend.application.agent.expert_agent.service.policy_filter_resolver import (
    merge_policy_filters,
    normalize_policy_filters,
)


@dataclass(frozen=True, slots=True)
class RetrievalProfile:
    """Retrieval constraints associated with an answer requirement."""

    policy_domain: tuple[str, ...] = ()
    content_type: tuple[str, ...] = ()
    jurisdiction: tuple[str, ...] = ()
    can_cite_as_policy_basis: bool = True

    def as_filters(self) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        if self.policy_domain:
            filters["policy_domain"] = list(self.policy_domain)
        if self.content_type:
            filters["content_type"] = list(self.content_type)
        if self.jurisdiction:
            filters["jurisdiction"] = list(self.jurisdiction)
        filters["can_cite_as_policy_basis"] = self.can_cite_as_policy_basis
        return filters


@dataclass(frozen=True, slots=True)
class ExtractionProfile:
    """Extraction handler and expected fact schema for a slot."""

    fact_schema: str
    extractor_id: str


@dataclass(frozen=True, slots=True)
class SlotDefinition:
    """Semantic answer slot, separated from retrieval and extraction details."""

    slot_id: str
    label: str
    question_patterns: tuple[str, ...]
    retrieval_profile: RetrievalProfile
    extraction_profile: ExtractionProfile
    coverage_terms: tuple[str, ...] = field(default_factory=tuple)
    broad: bool = False
    scenario_id: str = ""
    required_fields: tuple[str, ...] = field(default_factory=tuple)
    answer_action: str = "answer"


SLOT_REGISTRY: dict[str, SlotDefinition] = {
    "required_materials": SlotDefinition(
        slot_id="required_materials",
        label="必要材料",
        question_patterns=(
            "必要材料",
            "申请材料",
            "提交哪些材料",
            "需要哪些材料",
            "核验哪些材料",
            "应核验哪些材料",
            "需要核验哪些材料",
            "补充哪些材料",
            "报销材料",
            "材料目录",
        ),
        retrieval_profile=RetrievalProfile(
            policy_domain=("manual_reimbursement", "remote_medical_manual_reimbursement"),
            content_type=("policy_text", "service_guide"),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="material_list",
            extractor_id="material_list",
        ),
        coverage_terms=("材料名称", "材料必要性", "申请材料目录", "必要", "处方", "费用清单", "收据"),
        required_fields=("必要材料清单",),
        answer_action="list",
    ),
    "foreign_treatment_manual_reimbursement": SlotDefinition(
        slot_id="foreign_treatment_manual_reimbursement",
        label="外埠就医手工报销材料",
        question_patterns=("外埠就医", "外地就医", "易地安置"),
        retrieval_profile=RetrievalProfile(
            policy_domain=("manual_reimbursement",),
            content_type=("table_row", "policy_text"),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="scenario_material_rule",
            extractor_id="manual_reimbursement_scenario",
        ),
        coverage_terms=("外埠", "诊疗证明", "处方底方", "费用清单", "费用收据", "申报结算明细表"),
        scenario_id="foreign_treatment",
        required_fields=("诊疗证明", "处方底方", "费用清单", "费用收据", "申报结算明细表"),
        answer_action="scenario_rule",
    ),
    "account_settlement_voucher": SlotDefinition(
        slot_id="account_settlement_voucher",
        label="定点医药机构记账结算凭证",
        question_patterns=("记账结算", "定点医药机构", "定点医疗机构", "定点零售药店"),
        retrieval_profile=RetrievalProfile(
            policy_domain=("manual_reimbursement",),
            content_type=("table_row", "policy_text"),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="scenario_material_rule",
            extractor_id="manual_reimbursement_scenario",
        ),
        coverage_terms=("定点医疗机构", "定点零售药店", "个人帐户", "个人账户", "记账", "门急诊", "审核结算凭证"),
        scenario_id="account_settlement",
        required_fields=("个人账户支付", "记账结算", "门急诊费用审核结算凭证", "区县医保中心结算"),
        answer_action="scenario_rule",
    ),
    "emergency_manual_reimbursement_materials": SlotDefinition(
        slot_id="emergency_manual_reimbursement_materials",
        label="急诊就医手工报销材料",
        question_patterns=("急诊就医", "没带医保凭证", "未出示社保卡"),
        retrieval_profile=RetrievalProfile(
            policy_domain=("manual_reimbursement", "remote_medical_manual_reimbursement", "emergency"),
            content_type=("table_row", "policy_text", "faq"),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="scenario_material_rule",
            extractor_id="manual_reimbursement_scenario",
        ),
        coverage_terms=("急诊", "全额垫付", "收据", "处方", "诊断证明", "手工报销", "社保所"),
        scenario_id="emergency_visit",
        required_fields=("全额垫付", "收据", "处方", "诊断证明", "手工报销路径"),
        answer_action="scenario_rule",
    ),
    "statutory_processing_time": SlotDefinition(
        slot_id="statutory_processing_time",
        label="法定办结时限",
        question_patterns=("法定办结时限", "承诺办结时限", "办结时限", "办理时限", "多少工作日", "多久办结"),
        retrieval_profile=RetrievalProfile(
            policy_domain=("manual_reimbursement",),
            content_type=("policy_text", "service_guide"),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="processing_time",
            extractor_id="processing_time",
        ),
        coverage_terms=("法定办结时限", "承诺办结时限", "办理时限", "工作日", "审查", "决定"),
        required_fields=("法定办结时限",),
        answer_action="single_field",
    ),
    "drug_catalog": SlotDefinition(
        slot_id="drug_catalog",
        label="医保药品目录",
        question_patterns=(
            "药品",
            "药物",
            "用药",
            "药品目录",
            "医保药品",
            "甲类",
            "乙类",
            "限定支付",
            "药品编码",
            "目录编号",
        ),
        retrieval_profile=RetrievalProfile(
            policy_domain=("drug_catalog",),
            content_type=("table_row", "policy_text"),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="drug_catalog_row",
            extractor_id="drug_catalog",
        ),
        coverage_terms=("药品名称", "通用名", "目录编号", "医保类别", "限定支付范围", "本地支付比例"),
        required_fields=("药品名称", "医保类别", "目录编号"),
        answer_action="table_lookup",
    ),
    "medical_service_price": SlotDefinition(
        slot_id="medical_service_price",
        label="医疗服务项目价格",
        question_patterns=("医疗服务价格", "医疗服务项目", "诊疗项目", "计价单位", "收费标准", "CT", "ct"),
        retrieval_profile=RetrievalProfile(
            policy_domain=("medical_service_price",),
            content_type=("table_row", "policy_text"),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="medical_service_price_row",
            extractor_id="medical_service_price",
        ),
        coverage_terms=("项目名称", "编码", "计价单位", "收费标准", "内容说明", "医保类别"),
        required_fields=("项目名称", "编码", "计价单位", "收费标准"),
        answer_action="table_lookup",
    ),
    "remote_benefit_split": SlotDefinition(
        slot_id="remote_benefit_split",
        label="就医地目录与参保地待遇分工",
        question_patterns=(
            "就医地目录",
            "参保地待遇",
            "就医地支付范围",
            "参保地政策",
            "如何分工",
            "怎么分工",
            "直接结算支付规则",
            "医疗费用支付规则",
            "费用支付规则",
        ),
        retrieval_profile=RetrievalProfile(
            policy_domain=("remote_medical",),
            content_type=("policy_text",),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="remote_benefit_split",
            extractor_id="remote_benefit_split",
        ),
        coverage_terms=(
            "住院",
            "普通门诊",
            "门诊慢特病",
            "就医地",
            "参保地",
            "支付范围",
            "起付标准",
            "支付比例",
            "最高支付限额",
            "病种范围",
        ),
        scenario_id="direct_settlement",
        required_fields=(
            "就医地支付范围",
            "参保地起付标准",
            "参保地支付比例",
            "参保地最高支付限额",
        ),
        answer_action="explain_rule",
    ),
    "remote_self_pay_filing_manual_reimbursement": SlotDefinition(
        slot_id="remote_self_pay_filing_manual_reimbursement",
        label="补办备案后手工报销路径",
        question_patterns=("出院自费结算", "自费结算后", "补办备案", "补办备案手续", "补办备案后手工报销", "申请医保手工报销"),
        retrieval_profile=RetrievalProfile(
            policy_domain=("remote_medical", "remote_medical_manual_reimbursement"),
            content_type=("policy_text", "faq"),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="remote_self_pay_filing_manual_reimbursement",
            extractor_id="remote_self_pay_filing_manual_reimbursement",
        ),
        coverage_terms=("自费结算", "补办备案手续", "参保地规定", "医保手工报销"),
        scenario_id="self_pay_after_remote_filing",
        required_fields=("出院自费结算", "补办备案手续", "参保地规定", "医保手工报销"),
        answer_action="scenario_rule",
    ),
    "remote_emergency_observation_reimbursement": SlotDefinition(
        slot_id="remote_emergency_observation_reimbursement",
        label="异地急诊留观报销路径",
        question_patterns=("急诊留观直接结算", "异地急诊留观直接结算", "急诊留观报销路径", "留观费用直接结算", "住院标准报销"),
        retrieval_profile=RetrievalProfile(
            policy_domain=("remote_medical_manual_reimbursement",),
            content_type=("policy_text", "faq"),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="remote_emergency_observation_reimbursement",
            extractor_id="remote_emergency_observation_reimbursement",
        ),
        coverage_terms=("急诊留观", "暂不能实现异地直接结算", "票据", "相关报销材料", "社保所", "住院标准报销"),
        scenario_id="remote_emergency_observation",
        required_fields=("暂不能直接结算", "异地就医票据及相关报销材料", "单位或社保所提交", "区医保经办机构手工报销", "住院标准报销"),
        answer_action="scenario_rule",
    ),
    "remote_filing_institution_scope": SlotDefinition(
        slot_id="remote_filing_institution_scope",
        label="备案后就医机构范围",
        question_patterns=("备案成功后", "哪些机构就医", "哪些定点医药机构", "所有定点医药机构", "统筹地区"),
        retrieval_profile=RetrievalProfile(
            policy_domain=("remote_medical",),
            content_type=("policy_text", "faq"),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="remote_filing_institution_scope",
            extractor_id="policy_rule_span",
        ),
        coverage_terms=("备案", "就医地", "统筹地区", "所有定点医药机构", "就医结算"),
        scenario_id="direct_settlement",
        required_fields=("备案到就医地统筹地区", "统筹地区内所有定点医药机构", "按规定就医结算"),
        answer_action="list_scope",
    ),
    "benefit_params": SlotDefinition(
        slot_id="benefit_params",
        label="待遇参数",
        question_patterns=("起付线", "起付标准", "支付比例", "报销比例", "封顶线", "最高支付限额"),
        retrieval_profile=RetrievalProfile(
            policy_domain=("benefit", "remote_medical"),
            content_type=("policy_text",),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="benefit_parameter",
            extractor_id="benefit_params",
        ),
        coverage_terms=("起付标准", "起付线", "支付比例", "报销比例", "最高支付限额", "封顶线"),
        required_fields=("起付标准", "支付比例", "最高支付限额"),
        answer_action="single_field",
    ),
    "remote_filing": SlotDefinition(
        slot_id="remote_filing",
        label="异地备案与急诊例外",
        question_patterns=("备案", "补备案", "急诊例外", "急诊抢救", "视同备案", "视同已备案"),
        retrieval_profile=RetrievalProfile(
            policy_domain=("remote_medical", "remote_medical_manual_reimbursement", "emergency"),
            content_type=("policy_text", "faq", "service_guide"),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="policy_rule",
            extractor_id="policy_rule_span",
        ),
        coverage_terms=("备案", "补办", "补备案", "急诊抢救", "视同已备案", "手工报销"),
        required_fields=("备案规则",),
        answer_action="explain_rule",
    ),
    "consumable_payment_scope": SlotDefinition(
        slot_id="consumable_payment_scope",
        label="医用耗材支付范围",
        question_patterns=("医用耗材", "耗材支付范围", "医用耗材支付范围", "耗材目录", "支付办法", "先自负"),
        retrieval_profile=RetrievalProfile(
            policy_domain=("shanghai_payment_scope",),
            content_type=("policy_text", "table_row"),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="consumable_payment_scope",
            extractor_id="consumable_payment_scope",
        ),
        coverage_terms=("医用耗材", "耗材", "支付范围", "基金支付", "支付办法", "甲类", "乙类", "先自负"),
        required_fields=("支付范围", "支付办法"),
        answer_action="explain_rule",
    ),
    "special_disease_payment_condition": SlotDefinition(
        slot_id="special_disease_payment_condition",
        label="特殊疾病限定支付条件",
        question_patterns=("限定支付条件", "方可支付", "新增报销范围", "类风湿关节炎", "DMARDs", "风湿病专科医师"),
        retrieval_profile=RetrievalProfile(
            policy_domain=("special_disease_scope",),
            content_type=("policy_text", "table_row"),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="special_disease_condition",
            extractor_id="special_disease_condition",
        ),
        coverage_terms=("类风湿关节炎", "DMARDs", "3-6个月", "疾病活动度下降低于50%", "风湿病专科医师处方", "方可支付"),
        scenario_id="special_disease_payment",
        required_fields=("诊断明确", "传统DMARDs治疗3-6个月", "疾病活动度下降低于50%", "风湿病专科医师处方"),
        answer_action="condition_list",
    ),
    "benefit_policy": SlotDefinition(
        slot_id="benefit_policy",
        label="医保待遇与参保规则",
        question_patterns=(
            "城乡居民医保",
            "城乡老年人",
            "参保范围",
            "参保资格",
            "新生儿",
            "等待期",
            "待遇起始",
            "外埠户籍配偶",
            "家庭医生",
            "首诊转诊",
            "外省市目录",
        ),
        retrieval_profile=RetrievalProfile(
            policy_domain=("benefit",),
            content_type=("policy_text", "table_row"),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="benefit_policy",
            extractor_id="required_field_policy_rule",
        ),
        coverage_terms=(
            "城乡居民基本医疗保险",
            "城乡老年人",
            "参保人员范围",
            "新生儿",
            "待遇",
            "等待期",
            "外埠户籍配偶",
            "家庭医生签约",
            "首诊转诊",
            "外省市",
        ),
        required_fields=("待遇或参保规则",),
        answer_action="explain_rule",
    ),
    "chronic_long_prescription_policy": SlotDefinition(
        slot_id="chronic_long_prescription_policy",
        label="慢性病长处方政策",
        question_patterns=(
            "长期处方",
            "长处方",
            "慢性病",
            "慢病",
            "高血压",
            "糖尿病",
            "BJ-GBI",
            "医事服务费",
            "月度通报",
            "品种规格",
            "医联体",
        ),
        retrieval_profile=RetrievalProfile(
            policy_domain=("chronic_disease_long_prescription",),
            content_type=("policy_text",),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="chronic_long_prescription_policy",
            extractor_id="required_field_policy_rule",
        ),
        coverage_terms=(
            "长处方",
            "慢性病",
            "品种规格",
            "医联体",
            "BJ-GBI",
            "年终清算",
            "月度通报",
            "医事服务费",
            "高血压",
            "糖尿病",
            "按人头付费",
        ),
        required_fields=("慢性病长处方规则",),
        answer_action="explain_rule",
    ),
    "fund_supervision_policy": SlotDefinition(
        slot_id="fund_supervision_policy",
        label="医保基金监管规则",
        question_patterns=(
            "基金监管",
            "监督检查",
            "拒不配合",
            "暂停联网结算",
            "锁卡",
            "重点监督检查",
            "骗取基金",
            "涉嫌骗保",
            "不属于基金支付范围",
            "追回基金",
            "异常情形审核",
        ),
        retrieval_profile=RetrievalProfile(
            policy_domain=("fund_supervision",),
            content_type=("policy_text",),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="fund_supervision_policy",
            extractor_id="required_field_policy_rule",
        ),
        coverage_terms=(
            "监督检查",
            "基金使用",
            "不属于医疗保障基金支付范围",
            "拒不配合",
            "暂停联网结算",
            "骗取医疗保障基金",
            "服务协议",
            "异常情形审核",
        ),
        required_fields=("基金监管规则",),
        answer_action="explain_rule",
    ),
    "special_disease_filing_policy": SlotDefinition(
        slot_id="special_disease_filing_policy",
        label="门诊特殊病备案规则",
        question_patterns=(
            "特殊病备案",
            "特殊病种备案",
            "门诊特殊病备案",
            "门诊特殊疾病备案",
            "备案申报表",
            "医保办公室",
            "医保办",
            "病种名称",
            "中重度哮喘",
            "外埠户籍",
            "24个月",
        ),
        retrieval_profile=RetrievalProfile(
            policy_domain=("special_disease_filing",),
            content_type=("policy_text", "faq"),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="special_disease_filing_policy",
            extractor_id="required_field_policy_rule",
        ),
        coverage_terms=(
            "特殊病种备案申报表",
            "定点医院",
            "医保办公室",
            "参保区医疗保险经办机构",
            "住院期间",
            "出院手续",
            "外埠户籍",
            "连续缴纳",
            "中重度哮喘",
            "备案名称",
        ),
        required_fields=("特殊病备案规则",),
        answer_action="explain_rule",
    ),
    "special_disease_scope_policy": SlotDefinition(
        slot_id="special_disease_scope_policy",
        label="门诊特殊疾病范围",
        question_patterns=(
            "特殊疾病范围",
            "门诊特殊疾病范围",
            "新增病种",
            "重性精神病",
            "肺动脉高压",
            "耐多药结核",
            "尼曼匹克病",
            "特发性肺纤维化",
            "未备案",
            "选定特殊病种定点医疗机构",
        ),
        retrieval_profile=RetrievalProfile(
            policy_domain=("special_disease_scope",),
            content_type=("policy_text",),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="special_disease_scope_policy",
            extractor_id="required_field_policy_rule",
        ),
        coverage_terms=(
            "门诊特殊疾病范围",
            "重性精神病",
            "肺动脉高压",
            "耐多药结核",
            "C型尼曼匹克病",
            "中重度过敏性哮喘",
            "特发性肺纤维化",
            "报销范围",
            "备案审核",
            "不纳入",
        ),
        required_fields=("特殊疾病范围规则",),
        answer_action="explain_rule",
    ),
    "remote_settlement_management": SlotDefinition(
        slot_id="remote_settlement_management",
        label="跨省异地就医结算管理",
        question_patterns=(
            "银行手续费",
            "银行票据",
            "预付金",
            "黄色预警",
            "红色预警",
            "紧急调增",
            "费用协查",
            "一次性跨省住院",
            "国家跨省异地就医管理子系统",
        ),
        retrieval_profile=RetrievalProfile(
            policy_domain=("remote_medical",),
            content_type=("policy_text",),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="remote_settlement_management",
            extractor_id="required_field_policy_rule",
        ),
        coverage_terms=(
            "银行手续费",
            "银行票据工本费",
            "不得从基金中列支",
            "预付金",
            "黄色预警",
            "红色预警",
            "紧急调增",
            "费用协查",
            "国家跨省异地就医管理子系统",
        ),
        required_fields=("异地结算管理规则",),
        answer_action="explain_rule",
    ),
    "shanghai_service_facility_scope": SlotDefinition(
        slot_id="shanghai_service_facility_scope",
        label="上海医疗服务设施支付范围",
        question_patterns=(
            "医疗服务设施",
            "住院床位费",
            "急诊观察室床位费",
            "床位费",
            "实施期限",
            "有效期",
        ),
        retrieval_profile=RetrievalProfile(
            policy_domain=("shanghai_payment_scope",),
            content_type=("policy_text",),
            jurisdiction=("shanghai",),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="shanghai_service_facility_scope",
            extractor_id="required_field_policy_rule",
        ),
        coverage_terms=(
            "医疗服务设施",
            "住院床位费",
            "急诊观察室床位费",
            "基金支付范围",
            "有效期",
            "2026年7月31日",
        ),
        required_fields=("上海医疗服务设施范围规则",),
        answer_action="explain_rule",
    ),
    "negotiated_drug_double_channel": SlotDefinition(
        slot_id="negotiated_drug_double_channel",
        label="谈判药品与双通道规则",
        question_patterns=(
            "协议期内谈判药品",
            "谈判药品",
            "双通道",
            "电子处方",
            "一品两规",
            "药占比",
            "总额限制",
        ),
        retrieval_profile=RetrievalProfile(
            policy_domain=("drug_catalog", "shanghai_payment_scope"),
            content_type=("policy_text",),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="negotiated_drug_double_channel",
            extractor_id="required_field_policy_rule",
        ),
        coverage_terms=(
            "协议期内谈判药品",
            "乙类",
            "双通道",
            "电子处方",
            "一品两规",
            "药占比",
            "总额限制",
            "纳入基金支付范围",
        ),
        required_fields=("谈判药品或双通道规则",),
        answer_action="explain_rule",
    ),
    "designated_institution": SlotDefinition(
        slot_id="designated_institution",
        label="定点机构或药店状态",
        question_patterns=("定点机构", "定点医院", "定点药店", "机构编码", "药店编码", "定点状态"),
        retrieval_profile=RetrievalProfile(
            policy_domain=("designated_institution",),
            content_type=("table_row",),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="designated_institution_row",
            extractor_id="designated_institution",
        ),
        coverage_terms=("机构名称", "药店名称", "编码", "地址", "状态", "定点"),
        required_fields=("机构名称", "编码", "状态"),
        answer_action="table_lookup",
    ),
    "manual_reimbursement": SlotDefinition(
        slot_id="manual_reimbursement",
        label="手工报销",
        question_patterns=("手工报销", "零星报销", "外埠就医", "报销路径"),
        retrieval_profile=RetrievalProfile(
            policy_domain=("manual_reimbursement", "remote_medical_manual_reimbursement"),
            content_type=("policy_text", "service_guide"),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="policy_rule",
            extractor_id="policy_rule_span",
        ),
        coverage_terms=("手工报销", "零星报销", "费用清单", "专用收据", "审核结算"),
        required_fields=("手工报销规则",),
        answer_action="explain_rule",
        broad=True,
    ),
    "policy_basis": SlotDefinition(
        slot_id="policy_basis",
        label="政策依据",
        question_patterns=("政策", "依据", "规则", "规定"),
        retrieval_profile=RetrievalProfile(
            content_type=("policy_text", "table_row", "service_guide", "faq"),
        ),
        extraction_profile=ExtractionProfile(
            fact_schema="policy_rule",
            extractor_id="policy_rule_span",
        ),
        coverage_terms=("规定", "应当", "可以", "按照", "执行"),
        required_fields=("政策依据",),
        answer_action="explain_rule",
        broad=True,
    ),
}

ALIASED_SLOT_IDS = {
    "emergency_exception": "remote_filing",
    "filing_rule": "remote_filing",
    "remote_medical": "remote_settlement_management",
    "benefit": "benefit_policy",
    "fund_supervision": "fund_supervision_policy",
    "chronic_disease_long_prescription": "chronic_long_prescription_policy",
    "special_disease_filing": "special_disease_filing_policy",
    "special_disease_scope": "special_disease_scope_policy",
    "drug_price_reference": "policy_basis",
    "remote_emergency_observation": "remote_emergency_observation_reimbursement",
    "emergency_observation": "remote_emergency_observation_reimbursement",
    "self_pay_after_filing": "remote_self_pay_filing_manual_reimbursement",
    "double_channel": "negotiated_drug_double_channel",
    "negotiated_drug": "negotiated_drug_double_channel",
}


def normalize_answer_slot_id(slot_id: str) -> str:
    """Normalize a planner-provided slot id to a known registry slot."""

    raw = str(slot_id or "").strip()
    if not raw:
        return ""
    normalized = ALIASED_SLOT_IDS.get(raw, raw)
    return normalized if normalized in SLOT_REGISTRY else ""


def normalize_answer_slot_ids(value: Any) -> list[str]:
    """Normalize answer slot ids from LLM or deterministic plans."""

    if value in (None, "", [], {}):
        return []
    if isinstance(value, dict):
        raw_items: list[Any] = [value]
    elif isinstance(value, str):
        raw_items = [item.strip() for item in re.split(r"[,，\s]+", value) if item.strip()]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = [value]

    slot_ids: list[str] = []
    for item in raw_items:
        if isinstance(item, dict):
            raw = item.get("slot_id") or item.get("id") or item.get("name")
        else:
            raw = item
        slot_id = normalize_answer_slot_id(str(raw or ""))
        if slot_id:
            slot_ids.append(slot_id)
    return list(dict.fromkeys(slot_ids))


def slot_label(slot_id: str) -> str:
    definition = SLOT_REGISTRY.get(slot_id)
    return definition.label if definition is not None else slot_id


def resolve_answer_requirements(
    *,
    user_question: str,
    question_slots: list[dict[str, Any]],
    filters: dict[str, Any],
) -> list[AnswerRequirement]:
    """Resolve concrete answer requirements from question and existing slots."""

    normalized_filters = normalize_policy_filters(filters)
    slot_ids: list[str] = []
    slot_overrides: dict[str, dict[str, Any]] = {}
    for item in question_slots:
        if not isinstance(item, dict):
            continue
        slot_id = str(item.get("slot_id") or "")
        if not slot_id:
            continue
        normalized_slot_id = normalize_answer_slot_id(slot_id)
        if normalized_slot_id:
            slot_ids.append(normalized_slot_id)
            slot_overrides.setdefault(normalized_slot_id, item)

    compact_question = re.sub(r"\s+", "", str(user_question or ""))
    for slot_id, definition in SLOT_REGISTRY.items():
        if _matches_question(definition.question_patterns, compact_question):
            slot_ids.append(slot_id)

    if "special_disease_payment_condition" in slot_ids and not any(
        token in compact_question
        for token in ("医保类别", "目录编号", "药品编码", "本地支付比例", "支付比例是多少")
    ):
        slot_ids = [slot_id for slot_id in slot_ids if slot_id != "drug_catalog"]
    if "drug_catalog" in slot_ids and "benefit_params" in slot_ids and any(
        token in compact_question
        for token in ("医保类别", "目录编号", "药品目录", "本地支付比例")
    ) and not any(token in compact_question for token in ("起付", "封顶", "最高支付限额", "待遇参数")):
        slot_ids = [slot_id for slot_id in slot_ids if slot_id != "benefit_params"]

    slot_ids = _prefer_concrete_slots(slot_ids)
    requirements: list[AnswerRequirement] = []
    for index, slot_id in enumerate(slot_ids[:12], start=1):
        definition = SLOT_REGISTRY.get(slot_id)
        if definition is None:
            continue
        filters_for_slot = merge_policy_filters(
            normalized_filters,
            definition.retrieval_profile.as_filters(),
        )
        override = slot_overrides.get(slot_id, {})
        override_fields = override.get("required_fields") if isinstance(override, dict) else None
        required_fields = (
            [
                str(item).strip() for item in override_fields
                if str(item).strip()
            ]
            if isinstance(override_fields, list)
            else _required_fields_for_question(definition, user_question)
        )
        requirements.append(
            AnswerRequirement(
                requirement_id=f"req_{index}",
                slot_id=definition.slot_id,
                label=definition.label,
                question_span=_question_span_for_slot(user_question, definition),
                required=True,
                scenario_id=definition.scenario_id,
                required_fields=required_fields,
                answer_action=definition.answer_action,
                filters=filters_for_slot,
                fact_schema=definition.extraction_profile.fact_schema,
                extractor_id=definition.extraction_profile.extractor_id,
            )
        )
    if not requirements:
        definition = SLOT_REGISTRY["policy_basis"]
        requirements.append(
            AnswerRequirement(
                requirement_id="req_1",
                slot_id="policy_basis",
                label=definition.label,
                question_span=str(user_question or "")[:240],
                required=True,
                scenario_id=definition.scenario_id,
                required_fields=list(definition.required_fields),
                answer_action=definition.answer_action,
                filters=normalized_filters,
                fact_schema=definition.extraction_profile.fact_schema,
                extractor_id=definition.extraction_profile.extractor_id,
            )
        )
    return requirements


def slot_coverage_terms(slot_id: str) -> tuple[str, ...]:
    definition = SLOT_REGISTRY.get(slot_id)
    return definition.coverage_terms if definition is not None else ()


def slot_policy_domains(slot_id: str) -> tuple[str, ...]:
    definition = SLOT_REGISTRY.get(slot_id)
    if definition is None:
        return ()
    return definition.retrieval_profile.policy_domain


def slot_required_fields(slot_id: str) -> tuple[str, ...]:
    definition = SLOT_REGISTRY.get(slot_id)
    return definition.required_fields if definition is not None else ()


def slot_scenario_id(slot_id: str) -> str:
    definition = SLOT_REGISTRY.get(slot_id)
    return definition.scenario_id if definition is not None else ""


def slot_answer_action(slot_id: str) -> str:
    definition = SLOT_REGISTRY.get(slot_id)
    return definition.answer_action if definition is not None else "answer"


def _matches_question(patterns: tuple[str, ...], compact_question: str) -> bool:
    return any(pattern and pattern in compact_question for pattern in patterns)


def _required_fields_for_question(definition: SlotDefinition, question: str) -> list[str]:
    if definition.slot_id == "remote_benefit_split" and _looks_like_remote_direct_settlement_payment_question(question):
        return [
            "直接结算费用范围",
            "就医地支付范围",
            "参保地起付标准",
            "参保地支付比例",
            "参保地最高支付限额",
            "门诊慢特病病种范围",
        ]
    compact = re.sub(r"\s+", "", str(question or ""))
    if definition.slot_id == "remote_settlement_management":
        fields: list[str] = []
        if any(token in compact for token in ("银行手续费", "银行票据", "工本费")):
            fields.append("银行费用不得列支基金")
        if any(token in compact for token in ("预付金", "黄色预警", "红色预警", "紧急调增")):
            fields.extend(["预付金黄色预警", "预付金红色预警", "紧急调增流程"])
        if any(token in compact for token in ("费用协查", "一次性跨省住院", "3万元", "三万元")):
            fields.append("费用协查信息")
        return fields or list(definition.required_fields)
    if definition.slot_id == "benefit_policy":
        fields = []
        if any(token in compact for token in ("城乡老年人", "参保范围", "参保资格")):
            fields.append("城乡老年人参保范围")
        if any(token in compact for token in ("新生儿", "等待期", "待遇起始")):
            fields.extend(["新生儿待遇起始", "待遇等待期"])
        if any(token in compact for token in ("外埠户籍配偶", "配偶", "申请材料")):
            fields.append("外埠户籍配偶参保材料")
        if any(token in compact for token in ("家庭医生", "首诊转诊", "转诊")):
            fields.append("家庭医生签约首诊转诊")
        if any(token in compact for token in ("外省市目录", "外省市", "目录标准")):
            fields.append("外省市医疗费用目录标准")
        return fields or list(definition.required_fields)
    if definition.slot_id == "chronic_long_prescription_policy":
        fields = []
        if any(token in compact for token in ("品种规格", "品规", "医联体", "用药衔接")):
            fields.append("慢病药品品种规格衔接")
        if any(token in compact for token in ("BJ-GBI", "医事服务费", "年终清算", "补偿")):
            fields.append("医事服务费损失补偿")
        if any(token in compact for token in ("月度通报", "考核评分", "评分")):
            fields.append("长处方月度通报")
        if any(token in compact for token in ("高血压", "糖尿病", "按人头付费")):
            fields.append("高血压糖尿病按人头付费")
        return fields or list(definition.required_fields)
    if definition.slot_id == "fund_supervision_policy":
        fields = []
        if any(token in compact for token in ("不属于基金支付范围", "不予支付", "追回")):
            fields.append("不属于基金支付范围处理")
        if any(token in compact for token in ("拒不配合", "暂停联网结算", "锁卡")):
            fields.append("拒不配合调查处置")
        if any(token in compact for token in ("骗取基金", "涉嫌骗保", "违法违规")):
            fields.append("骗取基金处理程序")
        return fields or list(definition.required_fields)
    if definition.slot_id == "special_disease_filing_policy":
        fields = []
        if any(token in compact for token in ("本市就医", "异地就医", "医保办公室", "医保办", "经办机构")):
            fields.append("特殊病备案办理路径")
        if any(token in compact for token in ("住院期间", "出院手续")):
            fields.append("住院期间不得备案")
        if any(token in compact for token in ("外埠户籍", "24个月", "连续缴纳")):
            fields.append("外埠户籍特殊病备案条件")
        if any(token in compact for token in ("病种名称", "中重度哮喘", "名称调整")):
            fields.append("特殊病备案名称调整")
        return fields or list(definition.required_fields)
    if definition.slot_id == "special_disease_scope_policy":
        fields = []
        if any(token in compact for token in ("新增病种", "重性精神病", "肺动脉高压", "尼曼匹克", "纤维化")):
            fields.append("新增门诊特殊疾病病种")
        if any(token in compact for token in ("报销范围", "支付范围", "不纳入")):
            fields.append("门诊特殊疾病报销范围")
        if any(token in compact for token in ("备案审核", "选定特殊病种定点医疗机构", "未备案")):
            fields.append("备案审核后享受待遇")
        return fields or list(definition.required_fields)
    if definition.slot_id == "shanghai_service_facility_scope":
        fields = []
        if any(token in compact for token in ("住院床位费", "床位费")):
            fields.append("住院床位费纳入范围")
        if "急诊观察室床位费" in compact or "急诊观察" in compact:
            fields.append("急诊观察室床位费纳入范围")
        if any(token in compact for token in ("有效期", "实施期限", "2026年7月31日")):
            fields.append("政策有效期")
        return fields or list(definition.required_fields)
    if definition.slot_id == "negotiated_drug_double_channel":
        fields = []
        if "协议期内谈判药品" in compact or "谈判药品" in compact:
            fields.append("协议期内谈判药品乙类管理")
        if any(token in compact for token in ("双通道", "电子处方", "一品两规", "药占比", "总额限制")):
            fields.append("双通道药品供应约束")
        return fields or list(definition.required_fields)
    return list(definition.required_fields)


def _looks_like_remote_direct_settlement_payment_question(question: str) -> bool:
    compact = re.sub(r"\s+", "", str(question or ""))
    if not compact:
        return False
    return (
        any(token in compact for token in ("跨省", "异地就医", "异地"))
        and "直接结算" in compact
        and any(
            token in compact
            for token in ("医疗费用支付规则", "费用支付规则", "支付规则", "支付口径", "怎样支付", "如何支付")
        )
    )


def _prefer_concrete_slots(slot_ids: list[str]) -> list[str]:
    deduped = list(dict.fromkeys(slot_ids))
    scenario_slot_ids = {
        "foreign_treatment_manual_reimbursement",
        "account_settlement_voucher",
        "emergency_manual_reimbursement_materials",
        "remote_self_pay_filing_manual_reimbursement",
        "remote_emergency_observation_reimbursement",
    }
    if len(scenario_slot_ids.intersection(deduped)) >= 2:
        deduped = [
            slot_id for slot_id in deduped
            if slot_id not in {"required_materials", "manual_reimbursement", "remote_filing"}
        ]
    if "remote_emergency_observation_reimbursement" in deduped:
        deduped = [
            slot_id for slot_id in deduped
            if slot_id not in {"emergency_manual_reimbursement_materials", "remote_filing"}
        ]
    if "remote_self_pay_filing_manual_reimbursement" in deduped:
        deduped = [
            slot_id for slot_id in deduped
            if slot_id not in {"remote_filing", "manual_reimbursement"}
        ]
    if "remote_filing_institution_scope" in deduped:
        deduped = [
            slot_id for slot_id in deduped
            if slot_id != "remote_filing"
        ]
    if "remote_benefit_split" in deduped:
        deduped = [
            slot_id for slot_id in deduped
            if slot_id not in {"remote_medical", "benefit_params", "policy_basis"}
        ]
    if "special_disease_filing_policy" in deduped:
        deduped = [
            slot_id for slot_id in deduped
            if slot_id != "remote_filing"
        ]
    if "chronic_long_prescription_policy" in deduped and "drug_catalog" in deduped:
        deduped = [
            slot_id for slot_id in deduped
            if slot_id != "drug_catalog"
        ]
    if "benefit_policy" in deduped and "drug_catalog" in deduped:
        deduped = [
            slot_id for slot_id in deduped
            if slot_id != "drug_catalog"
        ]
    if "shanghai_service_facility_scope" in deduped:
        deduped = [
            slot_id for slot_id in deduped
            if slot_id not in {"medical_service_price", "consumable_payment_scope"}
        ]
    concrete = [
        slot_id for slot_id in deduped
        if not (SLOT_REGISTRY.get(slot_id).broad if SLOT_REGISTRY.get(slot_id) else False)
    ]
    if concrete:
        keep_broad = [
            slot_id for slot_id in deduped
            if slot_id not in {"manual_reimbursement", "policy_basis"}
        ]
        deduped = list(dict.fromkeys(keep_broad))
    return [slot_id for slot_id in deduped if slot_id in SLOT_REGISTRY]


def _question_span_for_slot(question: str, definition: SlotDefinition) -> str:
    return str(question or definition.label)[:240]
