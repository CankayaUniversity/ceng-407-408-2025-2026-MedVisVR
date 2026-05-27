export const BRAIN_PROMPTS = [
  "What is the total tumor volume?",
  "What is the enhancing tumor volume?",
  "What is the edema volume?",
  "Where is the tumor located?",
  "Which lobe is involved?",
  "How deep is the tumor from the cortical surface?",
  "What is the midline shift?",
  "Is there ventricular compression?",
  "Is there sulcal effacement?",
  "How close is the tumor to the motor cortex?",
  "How close is the tumor to the brainstem?",
  "Generate a BT-RADS structured report",
  "What is the dominant lesion component?",
  "Is this multifocal disease or a single dominant process?",
  "What are the key differential diagnoses?",
  "What are the surgical planning considerations?",
  "What is the herniation risk tier?",
  "Summarise everything we have discussed in this session so far.",
];

export const CT_PROMPTS = [
  "What is the total tumor volume?",
  "How many lesions were detected?",
  "What is the largest lesion and where is it located?",
  "Compare enhancing vs non-enhancing tumor volumes.",
  "What are the top limitations of this case-level analysis?",
];

export const SAMPLE_PROMPTS = CT_PROMPTS;
