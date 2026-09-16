"""人工初审与案件工作笔记 API 集成测试。"""


def test_add_case_note_and_return_it_in_case_detail(client, generated_case_id):
    note_response = client.post(
        f"/api/cases/{generated_case_id}/notes",
        json={
            "author": "审核员A",
            "source": "assistant",
            "content": "经人工核验，助手摘录的信息与当前申报项目一致。",
        },
    )
    detail_response = client.get(f"/api/cases/{generated_case_id}")

    assert note_response.status_code == 200
    assert detail_response.status_code == 200
    note = note_response.json()
    detail = detail_response.json()
    assert note["note_id"].startswith("NOTE-")
    assert note["source"] == "assistant"
    assert detail["notes"][0]["content"] == note["content"]


def test_add_case_note_with_material_metadata(client, generated_case_id):
    note_response = client.post(
        f"/api/cases/{generated_case_id}/notes",
        json={
            "author": "审核员A",
            "source": "external",
            "content": "已核验门诊费用明细，费用结构仍需结合申报项目复核。",
            "materials": [
                {
                    "name": "门诊费用明细",
                    "material_type": "费用清单",
                    "source": "审核人员登记",
                    "verification_status": "已核验",
                    "remark": "与申报项目费用结构核验相关。",
                }
            ],
        },
    )
    detail_response = client.get(f"/api/cases/{generated_case_id}")

    assert note_response.status_code == 200
    note = note_response.json()
    detail = detail_response.json()
    assert note["materials"][0]["name"] == "门诊费用明细"
    assert note["materials"][0]["verification_status"] == "已核验"
    assert detail["notes"][0]["materials"][0]["material_type"] == "费用清单"


def test_note_does_not_change_review_status_or_workflow(client, generated_case_id):
    response = client.post(
        f"/api/cases/{generated_case_id}/notes",
        json={
            "author": "审核员",
            "source": "manual",
            "content": "待进一步核验费用明细材料。",
        },
    )
    cases = client.get("/api/cases").json()
    workflow = client.get(f"/api/cases/{generated_case_id}/workflow").json()

    assert response.status_code == 200
    case = next(item for item in cases if item["case_id"] == generated_case_id)
    assert case["review_status"] == "pending"
    assert workflow["current_step"] == "initial_review"
    assert workflow["steps"][-1]["status"] == "pending"


def test_delete_note_removes_it_from_case_detail(client, generated_case_id):
    note_response = client.post(
        f"/api/cases/{generated_case_id}/notes",
        json={
            "author": "审核员",
            "source": "manual",
            "content": "该笔记录入有误，需要删除。",
        },
    )
    note_id = note_response.json()["note_id"]

    delete_response = client.delete(
        f"/api/cases/{generated_case_id}/notes/{note_id}"
    )
    detail = client.get(f"/api/cases/{generated_case_id}").json()

    assert delete_response.status_code == 200
    deleted = delete_response.json()
    assert deleted["note_id"] == note_id
    assert detail["notes"] == []


def test_legacy_void_note_endpoint_also_removes_it(client, generated_case_id):
    note_response = client.post(
        f"/api/cases/{generated_case_id}/notes",
        json={
            "author": "审核员",
            "source": "manual",
            "content": "兼容旧作废入口。",
        },
    )
    note_id = note_response.json()["note_id"]

    void_response = client.post(
        f"/api/cases/{generated_case_id}/notes/{note_id}/void"
    )
    detail = client.get(f"/api/cases/{generated_case_id}").json()

    assert void_response.status_code == 200
    assert void_response.json()["note_id"] == note_id
    assert detail["notes"] == []


def test_delete_note_rejected_after_review_submitted(client, generated_case_id):
    note_response = client.post(
        f"/api/cases/{generated_case_id}/notes",
        json={
            "author": "审核员",
            "source": "manual",
            "content": "初审前记录。",
        },
    )
    client.post(
        f"/api/cases/{generated_case_id}/review",
        json={
            "reviewer": "审核员A",
            "decision": "常规处理",
            "reason": "已完成初审。",
        },
    )

    response = client.delete(
        f"/api/cases/{generated_case_id}/notes/"
        f"{note_response.json()['note_id']}"
    )

    assert response.status_code == 422
    assert "人工初审已提交" in str(response.json()["detail"])


def test_note_rejects_forbidden_fields_and_unknown_case(client, generated_case_id):
    forbidden = client.post(
        f"/api/cases/{generated_case_id}/notes",
        json={
            "author": "审核员",
            "source": "external",
            "content": "材料中包含 RES 字段。",
        },
    )
    unknown = client.post(
        "/api/cases/CASE-UNKNOWN/notes",
        json={
            "author": "审核员",
            "source": "manual",
            "content": "测试记录。",
        },
    )

    assert forbidden.status_code == 422
    assert "RES" in str(forbidden.json()["detail"])
    assert unknown.status_code == 404


def test_note_rejects_forbidden_material_metadata(client, generated_case_id):
    response = client.post(
        f"/api/cases/{generated_case_id}/notes",
        json={
            "author": "审核员",
            "source": "external",
            "content": "登记外部材料核验情况。",
            "materials": [
                {
                    "name": "身份证材料",
                    "material_type": "其他",
                    "source": "审核人员登记",
                }
            ],
        },
    )

    assert response.status_code == 422
    assert "身份证" in str(response.json()["detail"])


def test_submit_review_with_attachment_metadata_and_return_it(client, generated_case_id):
    review_response = client.post(
        f"/api/cases/{generated_case_id}/review",
        json={
            "reviewer": "审核员A",
            "decision": "需补充材料",
            "reason": "需结合处方和费用明细进一步核验。",
            "attachments": [
                {
                    "name": "门诊材料",
                    "material_type": "业务材料",
                    "source": "审核人员登记",
                    "verification_status": "已核验",
                    "remark": "与费用结构核验相关。",
                    "file_name": "门诊材料.pdf",
                    "file_size": 204800,
                    "file_type": "application/pdf",
                    "file_last_modified": "2026/7/27 10:30:00",
                }
            ],
        },
    )
    detail = client.get(f"/api/cases/{generated_case_id}").json()
    workflow = client.get(f"/api/cases/{generated_case_id}/workflow").json()

    assert review_response.status_code == 200
    review = review_response.json()["review"]
    assert review["attachments"][0]["file_name"] == "门诊材料.pdf"
    assert detail["review"]["attachments"][0]["file_size"] == 204800
    assert workflow["current_step"] == "case_result"
    assert workflow["steps"][5]["status"] == "completed"
    assert workflow["steps"][6]["status"] == "conditional"
    assert workflow["steps"][7]["status"] == "conditional"
    assert workflow["steps"][8]["status"] == "recorded"


def test_submit_review_rejects_forbidden_attachment_metadata(client, generated_case_id):
    response = client.post(
        f"/api/cases/{generated_case_id}/review",
        json={
            "reviewer": "审核员A",
            "decision": "需补充材料",
            "reason": "测试",
            "attachments": [
                {
                    "name": "RES 字段截图",
                    "material_type": "其他",
                    "source": "审核人员登记",
                }
            ],
        },
    )

    assert response.status_code == 422
    assert "RES" in str(response.json()["detail"])


def test_submit_review_rejects_forbidden_local_file_metadata(client, generated_case_id):
    response = client.post(
        f"/api/cases/{generated_case_id}/review",
        json={
            "reviewer": "审核员A",
            "decision": "需补充材料",
            "reason": "测试",
            "attachments": [
                {
                    "name": "申报材料",
                    "material_type": "其他",
                    "source": "本地文件选择",
                    "file_name": "RES_export.pdf",
                    "file_size": 1024,
                    "file_type": "application/pdf",
                }
            ],
        },
    )

    assert response.status_code == 422
    assert "RES" in str(response.json()["detail"])
