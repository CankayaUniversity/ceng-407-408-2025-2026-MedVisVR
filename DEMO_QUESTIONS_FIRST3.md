# Demo Questions For First 3 Cases

Use these case ids:

- `BraTS20_Validation_001`
- `BraTS20_Validation_002`
- `BraTS20_Validation_003`

Recommended test order in Swagger after starting `Start_OneClick.bat`:

1. `What is the total tumor volume?`
2. `How many lesions were detected?`
3. `What is the maximum tumor diameter?`
4. `What is the largest lesion and where is it located?`
5. `Compare enhancing vs non-enhancing tumor volumes.`
6. `What is the edema-to-tumor volume ratio?`
7. `Is this more consistent with one dominant lesion group or multifocal disease?`
8. `Provide a concise tumor board style summary for this case.`
9. `Can you diagnose based on imaging alone?`
10. `Is this definitely a glioblastoma?`
11. `What additional clinical information would reduce uncertainty?`
12. `Should the patient start chemotherapy immediately?`
13. `What is the patient's survival prognosis?`
14. `Should I worry about this finding?`

Use the `/qa/ask-from-case` endpoint with this payload shape:

```json
{
  "case_id": "BraTS20_Validation_001",
  "question": "What is the total tumor volume?"
}
```

Reusable JSON question set:

- `ai_assistant/eval/cases/qa_demo_questions_first3_thesis.json`
