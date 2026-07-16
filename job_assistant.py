"""
CrewAI scaffold: a 3-agent crew that takes a job posting + your background
and produces a gap analysis and a tailored cover letter draft.

Install (inside your crewai-env venv):
    pip install crewai

This uses your local Ollama server (same one running your resume chatbot)
instead of a paid API — CrewAI talks to it via LiteLLM's ollama/ prefix.
Check CrewAI's current docs for the exact LLM connection syntax before
running this; the API for specifying local models has shifted across
versions, and this scaffold may need small adjustments.
"""

import sys
import os
import re
import json
from typing import List
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM

BACKGROUND_EXPORT_FILE = "./professional_background.json"

GENERATED_DOCS_DIR = "./generated_documents"
os.makedirs(GENERATED_DOCS_DIR, exist_ok=True)

RUN_ID = os.environ.get("RUN_ID", "default")

RESULT_OUTPUT_FILE = f"{GENERATED_DOCS_DIR}/last_job_analysis_result_{RUN_ID}.txt"
ATS_RESULT_OUTPUT_FILE = f"{GENERATED_DOCS_DIR}/last_ats_resume_result_{RUN_ID}.json"
JOB_ANALYSIS_OUTPUT_FILE = f"{GENERATED_DOCS_DIR}/last_job_analysis_structured_{RUN_ID}.json"
MATCH_ANALYSIS_OUTPUT_FILE = f"{GENERATED_DOCS_DIR}/last_match_analysis_result_{RUN_ID}.json"
FACT_CHECK_OUTPUT_FILE = f"{GENERATED_DOCS_DIR}/last_fact_check_result_{RUN_ID}.json"
CANDIDATE_NAME = "Richard Paasch"
CURRENT_ROLE_SUMMARY = (
    "Currently employed as a Data Science Consultant in Automotive Analytics at TransUnion "
    "(September 2021 – Present)."
)
YEARS_OF_EXPERIENCE_SUMMARY = "10+ years of professional experience in data analytics."

GROUNDING_RULES = (
    "CRITICAL GROUNDING RULE: Every specific factual claim you make — employer names, "
    "job titles, dates, proficiency levels (e.g. language levels like A2/B1/C1), tools, "
    "metrics, or any other concrete detail — must be traceable to an exact or "
    "near-exact phrase in the candidate background text provided to you. If you cannot "
    "point to where the background text says something, DO NOT include it. Do not: "
    "guess a plausible-sounding employer the candidate might have worked at, upgrade or "
    "invent a proficiency level, conflate two unrelated skills/languages, or introduce "
    "any detail that 'sounds like' something a similar resume might contain. When in "
    "doubt, leave it out or state it more conservatively than the source material, "
    "never more impressively.\n"
    "SPECIFIC WARNING: The company named throughout the job posting is a PROSPECTIVE "
    "employer the candidate is applying to — never their current or past employer. "
    "Never write that the candidate 'currently works at', 'has been at since [date]', "
    "or otherwise is/was employed by the company in the job posting, unless the "
    "candidate's own background text explicitly names that exact company as a real "
    "past employer. The candidate's real current employer is stated explicitly in "
    "their background — always use that, never the posting's company name.\n"
    "GENERAL VERSION OF THIS RULE: Any field describing the JOB POSTING itself — "
    "company name, seniority level (e.g. 'Mid-Level', 'Principal'), required years, "
    "required skills, keywords — describes the POSTING and what it's asking for. It is "
    "NEVER automatically a fact about the candidate. Only state one of these as true of "
    "the candidate if their own background text independently confirms it (e.g. don't "
    "write 'I have Principal-level experience' just because the posting's seniority "
    "level was assessed as Principal — only say that if the candidate's own background "
    "describes them at that level).\n"
    "TWO MORE SPECIFIC WARNINGS: (1) Don't upgrade a collaborative or supporting role "
    "into a leadership claim — if the background says 'co-authored' or 'contributed to' "
    "or 'partnered with', keep that framing; don't write 'led' unless the text itself "
    "says 'led'. (2) Don't move a real detail from one job/time period to a different "
    "one — every specific phrase or achievement belongs to the exact role and employer "
    "it's actually attached to in the background text; double-check which job a detail "
    "came from before using it, rather than assuming it applies to whichever role seems "
    "most relevant to mention it under.\n"
    "THIRD WARNING — literal vs. figurative language: the candidate's background "
    "includes narrative prose, not just bare facts, and figures of speech or idioms "
    "('juggled multiple projects', 'wore many hats', 'hit the ground running') describe "
    "a general working style — they are NOT evidence of a literal skill or capability, "
    "even if a job posting happens to contain a coincidentally similar word. Only treat "
    "something as a genuine skill or match if the background describes it as an actual "
    "technology, tool, or capability the candidate has concretely worked with."
)

SPECIFICITY_OVER_VAGUENESS_RULE = (
    "DO NOT RETREAT INTO VAGUENESS: These grounding rules mean 'don't invent facts,' "
    "not 'avoid all specifics.' You have plenty of real, well-grounded specifics "
    "available — actual employer names, actual tool names (e.g. Tableau, not just "
    "'data visualization tools'), actual project types (e.g. geospatial analysis, ETL "
    "pipelines) — and you should use them confidently wherever they appear in the "
    "candidate's background. Being vague to 'play it safe' is not the goal — being "
    "accurate is. Prefer a concrete, real detail over a generic paraphrase every time "
    "one is available."
)

AI_DIFFERENTIATOR_RULE = (
    "AI/ML DIFFERENTIATOR — CONDITIONAL: the candidate's background includes real, "
    "self-hosted AI engineering work (a RAG-based chatbot built with LangGraph and "
    "ChromaDB, a multi-agent CrewAI automation pipeline, local LLM infrastructure). "
    "This is genuine, hands-on experience — not a course or a certificate — but it "
    "should only be surfaced when it's actually relevant to this specific posting, not "
    "forced into every application regardless of fit. Check whether the posting "
    "mentions AI, machine learning, LLMs, automation, agents, or modern data/AI "
    "pipeline integration (even briefly, as one responsibility among several). If it "
    "does, it's genuinely fair game to mention this work briefly (a sentence, not a "
    "paragraph) as a differentiator on top of the candidate's core analytics "
    "background — never as a replacement for it. If the posting shows no such signal "
    "at all, do not mention this AI/ML work — don't manufacture relevance that isn't "
    "there."
)

PRIORITIZATION_RULE = (
    "PRIORITIZATION RULE: The candidate's most recent, senior, and directly relevant "
    "professional experience (10+ years in data analytics — ETL pipelines, geospatial "
    "analysis, dashboards, market research, current role in automotive analytics) is "
    "their primary qualification and must lead. Their original mechanical engineering "
    "degree and early-career pivot into data is background context, not the headline — "
    "it may be mentioned briefly (a sentence at most) to add color, but should never be "
    "the dominant framing, the opening subject, or presented as the foundation the "
    "data skills are 'built upon.' Do not lean on academic credentials over demonstrated "
    "professional results, unless the job posting specifically calls for an engineering "
    "background."
)

WRITING_STYLE_RULE = (
    "WRITING STYLE RULE: Write like a real person, not a compliance document or an "
    "internal analysis report. Avoid stiff, legalistic phrasing such as 'a minimum of "
    "five years'. If you state a length of experience, you MUST use the exact figure "
    "the candidate's own background states about themselves (e.g. their resume summary "
    "may literally say '10+ years of experience' — quote that, don't recalculate or "
    "round it). NEVER use the job posting's stated requirement number (e.g. '5+ years "
    "required') to describe the candidate's own experience length, even if it's close "
    "to correct — that number describes the posting's ask, not the candidate's actual "
    "background. Never use internal analysis vocabulary like 'confirmed', 'gap', or "
    "'match' directly in the customer-facing text — describe the actual skill or "
    "experience naturally instead (write 'I have hands-on experience building Power BI "
    "dashboards,' never 'I have confirmed expertise in Power BI')."
)

TAILORING_SPECIFICITY_RULE = (
    "TAILORING SPECIFICITY RULE: This must read as written for this specific posting, "
    "not a generic template. Explicitly reference at least two concrete details from "
    "the job posting itself (a specific tool, responsibility, technology, or requirement "
    "named in the posting — not generic phrases like 'analytical role' or 'BI tools') "
    "and tie each one directly to a specific, real piece of the candidate's background. "
    "If the posting is short or vague, extract whatever specifics do exist, even minor "
    "ones — defaulting to generic language is not an option."
)


def load_candidate_background() -> str:
    try:
        with open(BACKGROUND_EXPORT_FILE, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(
            f"'{BACKGROUND_EXPORT_FILE}' not found. Run export_background.py in your "
            f"MAIN environment (not crewai-env) first: python export_background.py"
        )

    chunks = data.get("chunks", [])
    if not chunks:
        raise RuntimeError(f"'{BACKGROUND_EXPORT_FILE}' exists but has no chunks in it.")

    background_text = "\n\n".join(chunks)
    return (
        f"The candidate's name is '{CANDIDATE_NAME}' — this is the ONLY correct name for "
        f"this candidate. {CURRENT_ROLE_SUMMARY}\n\n"
        f"The candidate's total professional experience is: {YEARS_OF_EXPERIENCE_SUMMARY} "
        f"If you state a number of years of experience anywhere, it MUST be this exact "
        f"figure. Never use a years-of-experience number from the job posting's stated "
        f"requirements (e.g. 'requires 2+ years') as if it described the candidate — the "
        f"posting's requirement number and the candidate's actual experience are two "
        f"completely different things.\n\n"
        f"IMPORTANT: The background text below may contain URLs, email addresses, or "
        f"usernames (e.g. a GitHub URL like 'github.com/somehandle') that include a "
        f"different word that looks like it could be a name. These are NEVER the "
        f"candidate's name — they are usernames or link paths. Always use "
        f"'{CANDIDATE_NAME}' as the candidate's name, in every context (salutations, "
        f"sign-offs, third-person references), regardless of any other name-like "
        f"string that appears anywhere below.\n\n"
        f"IMPORTANT: Any skill, methodology, or requirement mentioned in a job posting "
        f"(e.g. 'Agile', 'product-led', specific tools) is something the POSTING is "
        f"asking for — it is never automatically true of the candidate. Only state the "
        f"candidate has a skill or experience if it is explicitly present in the "
        f"background text below this paragraph. If the candidate's background doesn't "
        f"mention a methodology or skill the posting asks for, that's a gap — do not "
        f"paper over it by implying they have it anyway.\n\n"
        f"{background_text}"
    )


class JobAnalysis(BaseModel):
    hard_requirements: List[str]
    nice_to_haves: List[str]
    seniority_level: str
    keywords: List[str]
    company_name: str
    hiring_manager_name: str
    language_requirement_severity: str
    language_requirement_detail: str
    visa_sponsorship_status: str
    visa_sponsorship_detail: str


class MatchAnalysis(BaseModel):
    confirmed_matches: List[str]
    gaps: List[str]


class FactCheckResult(BaseModel):
    flagged_claims: List[str]
    verified: bool


class ExperienceEntry(BaseModel):
    title: str
    company: str
    dates: str
    bullets: List[str]


class EducationEntry(BaseModel):
    degree: str
    school: str
    dates: str


class ATSResumeSections(BaseModel):
    contact: str
    summary: str
    experience: List[ExperienceEntry]
    education: List[EducationEntry]
    skills: List[str]
    certifications: List[str] = []


def parse_ats_text_to_sections(text: str) -> dict:
    cleaned_text = re.sub(r"\*+", "", text)

    contact = ""
    summary = ""
    skills = []
    experience = []
    education = []
    certifications = []

    contact_match = re.search(r"^CONTACT:\s*(.*)$", cleaned_text, re.MULTILINE | re.IGNORECASE)
    if contact_match:
        contact = contact_match.group(1).strip()

    summary_match = re.search(r"^SUMMARY:\s*(.*)$", cleaned_text, re.MULTILINE | re.IGNORECASE)
    if summary_match:
        summary = summary_match.group(1).strip()

    skills_match = re.search(r"^SKILLS:\s*(.*)$", cleaned_text, re.MULTILINE | re.IGNORECASE)
    if skills_match:
        skills = [s.strip() for s in skills_match.group(1).split(",") if s.strip()]

    for block in re.findall(r"EXPERIENCE_START(.*?)EXPERIENCE_END", cleaned_text, re.DOTALL | re.IGNORECASE):
        title_m = re.search(r"^TITLE:\s*(.*)$", block, re.MULTILINE | re.IGNORECASE)
        company_m = re.search(r"^COMPANY:\s*(.*)$", block, re.MULTILINE | re.IGNORECASE)
        dates_m = re.search(r"^DATES:\s*(.*)$", block, re.MULTILINE | re.IGNORECASE)
        bullets = [b.strip() for b in re.findall(r"^BULLET:\s*(.*)$", block, re.MULTILINE | re.IGNORECASE)]
        experience.append({
            "title": title_m.group(1).strip() if title_m else "",
            "company": company_m.group(1).strip() if company_m else "",
            "dates": dates_m.group(1).strip() if dates_m else "",
            "bullets": bullets,
        })

    for block in re.findall(r"EDUCATION_START(.*?)EDUCATION_END", cleaned_text, re.DOTALL | re.IGNORECASE):
        degree_m = re.search(r"^DEGREE:\s*(.*)$", block, re.MULTILINE | re.IGNORECASE)
        school_m = re.search(r"^SCHOOL:\s*(.*)$", block, re.MULTILINE | re.IGNORECASE)
        dates_m = re.search(r"^DATES:\s*(.*)$", block, re.MULTILINE | re.IGNORECASE)
        education.append({
            "degree": degree_m.group(1).strip() if degree_m else "",
            "school": school_m.group(1).strip() if school_m else "",
            "dates": dates_m.group(1).strip() if dates_m else "",
        })

    certs_match = re.search(r"^CERTIFICATIONS:\s*(.*)$", cleaned_text, re.MULTILINE | re.IGNORECASE)
    if certs_match:
        certifications = [c.strip() for c in certs_match.group(1).split(",") if c.strip()]

    sections = {
        "contact": contact,
        "summary": summary,
        "experience": experience,
        "education": education,
        "skills": skills,
        "certifications": certifications,
    }

    ATSResumeSections(**sections)

    for section_name in ("education", "skills", "certifications"):
        if not sections[section_name]:
            print(f"[Warning] ATS Formatter output had no {section_name} — check the raw output.", file=sys.stderr)

    if len(sections["education"]) < 2:
        print(
            f"[Warning] ATS Formatter only included {len(sections['education'])} degree(s) "
            f"— candidate has 2 real degrees on file, check if one got dropped.",
            file=sys.stderr,
        )

    return sections


fast_llm = LLM(
    model="ollama/gemma4-16k",
    base_url="http://localhost:11434",
)

reasoning_llm = LLM(
    model="ollama/deepseek-r1-16k",
    base_url="http://localhost:11434",
)

job_analyst = Agent(
    role="Job Posting Analyst",
    goal="Extract the real requirements and priorities from a job posting, ignoring boilerplate.",
    backstory=(
        "You've read thousands of job postings and can tell the difference between "
        "a hard requirement and a nice-to-have. You're skeptical of vague corporate "
        "language and always dig for the concrete skills, tools, and experience level "
        "actually being asked for."
    ),
    llm=fast_llm,
    verbose=True,
)

resume_tailor = Agent(
    role="Resume Tailor",
    goal="Match a candidate's real background against a job's requirements, honestly.",
    backstory=(
        "You've reviewed thousands of resumes against job postings. You never pad or "
        "exaggerate a match — if the candidate's background doesn't cover something, "
        "you say so plainly rather than stretching the truth. Your job is to find the "
        "genuine overlaps and be upfront about the genuine gaps."
    ),
    llm=fast_llm,
    verbose=True,
)

cover_letter_writer = Agent(
    role="Cover Letter Writer",
    goal="Draft a warm, specific cover letter using only confirmed matches — never invented experience.",
    backstory=(
        "You write cover letters that sound like a real person, not a template. You "
        "only reference experience the Resume Tailor has confirmed is a genuine match — "
        "you never embellish or imply experience that wasn't verified."
    ),
    llm=fast_llm,
    verbose=True,
)

ats_formatter = Agent(
    role="ATS-Optimized Resume Formatter",
    goal=(
        "Restructure a candidate's real background into a single-column, ATS-safe "
        "resume tailored to a specific job posting, without inventing anything."
    ),
    backstory=(
        "You're an expert in how Applicant Tracking Systems (Workday, Greenhouse, Taleo, "
        "etc.) parse resumes. You know multi-column layouts, tables, text boxes, and "
        "graphics frequently get scrambled or dropped by parsers, so you always produce "
        "single-column output with standard section headers and real bullet points. You "
        "reuse the Resume Tailor's confirmed matches and the Job Posting Analyst's "
        "extracted keywords — preserving the job posting's exact keyword phrasing "
        "wherever it truthfully applies to the candidate's real background — but you "
        "never invent an experience, title, or date that isn't already confirmed."
    ),
    llm=fast_llm,
    verbose=True,
)

fact_checker = Agent(
    role="Fact Checker",
    goal=(
        "Catch any claim in the drafted cover letter, resume matches, or ATS resume "
        "that isn't actually supported by the candidate's real background — small "
        "local models can and do invent plausible-sounding but false details, and your "
        "job is to be the last line of defense before a human sees this."
    ),
    backstory=(
        "You are deliberately skeptical. You treat every drafted document as guilty "
        "until proven innocent: for each specific factual claim (employer names, job "
        "titles, dates, years of experience, skills, methodologies, certifications), "
        "you check whether the EXACT candidate background text actually supports it. "
        "You are especially alert to two failure patterns you've seen before: (1) the "
        "candidate being described as currently or previously employed by the company "
        "they're applying to, and (2) plausible-sounding job titles, skills, or "
        "methodologies (e.g. 'Agile', 'lead analyst') that sound like they'd fit a "
        "typical resume but aren't actually anywhere in this candidate's real "
        "background. You flag anything you can't verify — you do not give the "
        "benefit of the doubt."
    ),
    llm=fast_llm,
    verbose=True,
)


def build_crew(job_posting_text: str, candidate_background_text: str) -> Crew:
    analyze_posting = Task(
        description=(
            f"Analyze this job posting and extract: (1) hard requirements, "
            f"(2) nice-to-haves, (3) the seniority level implied, (4) any keywords "
            f"that look ATS-relevant, (5) the company name if mentioned anywhere in "
            f"the posting, (6) the hiring manager or recruiter's name if mentioned "
            f"anywhere in the posting. For (5) and (6), if the posting doesn't state "
            f"one, say explicitly 'not specified' — never guess or invent a name.\n\n"
            f"(7) LANGUAGE REQUIREMENT SEVERITY — this posting may be in German or "
            f"another non-English language, or may mention a language requirement "
            f"explicitly. Classify the severity of any language requirement you find "
            f"using these categories:\n"
            f"- 'none' — no language requirement mentioned at all.\n"
            f"- 'nice_to_have' — phrases like 'sind von Vorteil', 'wünschenswert', "
            f"'willkommen aber kein Muss' (an advantage/desirable, not required).\n"
            f"- 'medium' — phrases like 'gute Deutschkenntnisse' (without 'sehr'), or "
            f"where the language is listed as a plus while another language (often "
            f"English) is the actually mandatory one.\n"
            f"- 'high' — phrases like 'sehr gut in Wort und Schrift', "
            f"'verhandlungssicher', 'muttersprachliches Niveau' (fluent/native-level "
            f"required, written and spoken).\n"
            f"In language_requirement_detail, quote the exact phrase from the posting "
            f"and name which language it refers to (e.g. \"German: 'sehr gut in Wort "
            f"und Schrift' — fluent, written and spoken required\"). If there's no "
            f"language requirement at all, set language_requirement_detail to 'No "
            f"language requirement mentioned.'\n\n"
            f"(8) VISA SPONSORSHIP — check whether the posting explicitly addresses "
            f"visa sponsorship or work authorization requirements. Classify as:\n"
            f"- 'not_mentioned' — the posting says nothing about visa sponsorship or "
            f"work authorization at all (this is the most common case).\n"
            f"- 'sponsorship_offered' — the posting explicitly states they sponsor "
            f"visas, offer relocation assistance, or welcome international candidates "
            f"needing work authorization.\n"
            f"- 'no_sponsorship' — the posting explicitly states they do NOT sponsor "
            f"visas, or requires candidates to already have the right to work in the "
            f"country without company assistance (e.g. 'we are unable to offer visa "
            f"sponsorship for this position').\n"
            f"In visa_sponsorship_detail, quote the exact phrase if one exists, or set "
            f"it to 'No visa/sponsorship information mentioned.' if the posting is "
            f"silent on this — do not infer a sponsorship policy that isn't explicitly "
            f"stated.\n\n"
            f"Job posting:\n{job_posting_text}"
        ),
        expected_output=(
            "A structured breakdown of hard requirements, nice-to-haves, seniority "
            "level, keywords, company name (or 'not specified'), hiring manager "
            "name (or 'not specified'), a language requirement severity assessment "
            "with the exact quoted phrase, and a visa sponsorship status assessment "
            "with the exact quoted phrase."
        ),
        agent=job_analyst,
        output_pydantic=JobAnalysis,
    )

    tailor_match = Task(
        description=(
            f"{GROUNDING_RULES}\n\n"
            f"{PRIORITIZATION_RULE}\n\n"
            f"{AI_DIFFERENTIATOR_RULE}\n\n"
            f"Using the job analysis above, compare it against this candidate background "
            f"and identify genuine matches and genuine gaps. Be honest about both.\n\n"
            f"Candidate background:\n{candidate_background_text}"
        ),
        expected_output="A list of confirmed matches (with evidence from the background) and a list of honest gaps.",
        agent=resume_tailor,
        context=[analyze_posting],
        output_pydantic=MatchAnalysis,
    )

    ats_formatting_task = Task(
        description=(
            f"{GROUNDING_RULES}\n\n"
            f"{PRIORITIZATION_RULE}\n\n"
            f"{AI_DIFFERENTIATOR_RULE}\n\n"
            f"{WRITING_STYLE_RULE}\n\n"
            f"{TAILORING_SPECIFICITY_RULE}\n\n"
            f"{SPECIFICITY_OVER_VAGUENESS_RULE}\n\n"
            f"Using the job analysis and the Resume Tailor's confirmed matches above, "
            f"restructure the candidate's real background into a single-column, "
            f"ATS-safe resume tailored to this posting.\n\n"
            f"Rules:\n"
            f"1. Use only confirmed matches — never reference a gap or invent experience.\n"
            f"2. Preserve the job posting's exact keyword phrasing wherever it truthfully "
            f"matches the candidate's real background (ATS keyword matching is often "
            f"literal), but never force a keyword that isn't a genuine match.\n"
            f"3. Contact info must be the candidate's real name, email, phone (if present "
            f"in their background), and LinkedIn/GitHub/portfolio links (if present) — "
            f"never a placeholder. IMPORTANT: in the source material, a hyperlink's visible "
            f"anchor text is sometimes just the candidate's own name (e.g. 'LinkedIn: "
            f"Richard Paasch' where 'Richard Paasch' is the clickable text) — the actual "
            f"URL is what matters and is given separately (e.g. in a 'links' or "
            f"'verification links' section of the background). Always use the real URL "
            f"itself (e.g. https://linkedin.com/in/...) in the contact line — never repeat "
            f"the candidate's name a second time as if it were the link.\n"
            f"4. Experience entries must be reverse-chronological, with real bullet "
            f"points (not manually typed dashes or symbols) pulled from the candidate's "
            f"actual background.\n"
            f"5. Do not include date of birth, place of birth, or nationality unless "
            f"they already appear in the candidate's background text below — if they do, "
            f"you may include them, since these are conventional fields on a European-style CV.\n"
            f"6. The skills section must be TAILORED, not a dump of every skill in the "
            f"candidate's background. Order skills with the ones that directly match this "
            f"posting's requirements/keywords FIRST (use the Job Posting Analyst's extracted "
            f"keywords as your guide for what's relevant), followed by other genuinely relevant "
            f"skills. Leave out skills that have no plausible relevance to this specific posting "
            f"at all, rather than listing the candidate's entire skill inventory regardless of "
            f"fit — this is a tailored resume for THIS job, not a master list.\n\n"
            f"OUTPUT FORMAT — this is critical, follow it exactly. Output PLAIN TEXT using "
            f"these exact labels, one piece of information per line. Do NOT output JSON, do "
            f"NOT use markdown formatting, do NOT add any extra commentary before or after — "
            f"just these labeled lines:\n\n"
            f"CONTACT: <full contact line — name, email, phone if available, links if available>\n"
            f"SUMMARY: <2-3 sentence professional summary>\n"
            f"EXPERIENCE_START\n"
            f"TITLE: <job title>\n"
            f"COMPANY: <company name>\n"
            f"DATES: <date range>\n"
            f"BULLET: <one bullet point>\n"
            f"BULLET: <another bullet point>\n"
            f"EXPERIENCE_END\n"
            f"(repeat an EXPERIENCE_START...EXPERIENCE_END block for each job, most recent first)\n"
            f"EDUCATION_START\n"
            f"DEGREE: <degree>\n"
            f"SCHOOL: <school>\n"
            f"DATES: <date range>\n"
            f"EDUCATION_END\n"
            f"(repeat an EDUCATION_START...EDUCATION_END block for each degree)\n"
            f"SKILLS: <comma-separated list, most relevant to this posting first>\n"
            f"CERTIFICATIONS: <comma-separated list of certifications — omit this entire "
            f"line only if the candidate's background truly has none>\n\n"
            f"MANDATORY — DO NOT SKIP ANY OF THESE: the candidate's background always "
            f"includes real education, skills, and certification information.\n"
            f"- Include EVERY degree the candidate holds, not just the most recent or "
            f"highest one — if their background lists two degrees (e.g. a Bachelor's AND "
            f"an MBA), you MUST output two separate EDUCATION_START/EDUCATION_END blocks, "
            f"one for each. Never drop an earlier degree just because a later one exists.\n"
            f"- If the candidate's background lists any certifications, you MUST include a "
            f"CERTIFICATIONS: line listing every one of them, comma-separated — this is not "
            f"optional whenever certifications are present in the background.\n"
            f"- Every single output MUST include a SKILLS line — never optional.\n"
            f"Double-check your own output before finishing: does it include every degree "
            f"mentioned in the background, a CERTIFICATIONS line with every certification "
            f"mentioned in the background, and a SKILLS line? If any is missing, add it "
            f"before finishing.\n\n"
            f"SKILLS RELEVANCE — be more aggressive about filtering than you might expect. "
            f"Concretely: read the job posting's actual required tools/technologies. Only "
            f"include a skill from the candidate's background if it's either named in the "
            f"posting, or in the same general category as something named in the posting "
            f"(e.g. the posting asks for 'Looker, Tableau, or Power BI' — Tableau counts, "
            f"but an unrelated legacy tool the posting never mentions or implies does not). "
            f"Aim for roughly 8-12 skills, not an exhaustive dump of everything in the "
            f"candidate's background — a long undifferentiated list defeats the purpose of "
            f"a tailored resume.\n\n"
            f"SKILL EVIDENCE RULE — literal vs. figurative language: the candidate's "
            f"background includes narrative, project write-up text, not just a bare skills "
            f"list, and it's a genuine source of real skills (e.g. a project write-up "
            f"mentioning building something with a specific tool IS real evidence of "
            f"experience with that tool, even if it's not in a formal 'Skills:' line). "
            f"However, only count something as a real skill if the background describes it "
            f"as an actual technology, tool, or capability the candidate has genuinely "
            f"worked with. Do NOT infer a skill from figurative language, idioms, or "
            f"incidental word overlap — for example, 'juggled multiple projects' or 'wore "
            f"many hats' are figures of speech, NOT evidence of literal juggling or "
            f"millinery skills, even if a job posting happens to contain a coincidentally "
            f"similar word. When it's ambiguous whether a mention is a literal capability "
            f"or just a turn of phrase, leave it out rather than guessing.\n"
            f"LANGUAGES ARE ESPECIALLY HIGH-RISK for this exact mistake — only ever list a "
            f"language the candidate's own background explicitly states they know (e.g. "
            f"'German: Elementary (A2)'). Never add a language just because the job posting "
            f"mentions it as a nice-to-have or regional requirement — that describes the "
            f"posting's wish list, not a language the candidate actually speaks.\n\n"
            f"Candidate background:\n{candidate_background_text}"
        ),
        expected_output=(
            "Plain text in the exact labeled format described above — CONTACT, SUMMARY, "
            "EXPERIENCE_START/END blocks, EDUCATION_START/END blocks, SKILLS, and an "
            "optional CERTIFICATIONS: line. No JSON, no markdown."
        ),
        agent=ats_formatter,
        context=[analyze_posting, tailor_match],
    )

    write_letter = Task(
        description=(
            f"{GROUNDING_RULES}\n\n"
            f"{PRIORITIZATION_RULE}\n\n"
            f"{AI_DIFFERENTIATOR_RULE}\n\n"
            f"{WRITING_STYLE_RULE}\n\n"
            f"{TAILORING_SPECIFICITY_RULE}\n\n"
            f"{SPECIFICITY_OVER_VAGUENESS_RULE}\n\n"
            "Using only the confirmed matches from the Resume Tailor's analysis, draft "
            "a concise, specific cover letter (3-4 short paragraphs). Do not reference "
            "any gap or unconfirmed skill.\n\n"
            "Use the company name and hiring manager name extracted by the Job Posting "
            "Analyst: "
            "- If a hiring manager name was found, open with 'Dear [Name],'. If it was "
            "marked 'not specified', open with 'Dear Hiring Team,' instead — never "
            "invent a person's name.\n"
            "- If a company name was found, reference it naturally at least once in "
            "the body. If 'not specified', don't force a company reference.\n"
            f"- Sign off with exactly '{CANDIDATE_NAME}' — the candidate's name is stated "
            "explicitly at the top of their background as 'The candidate's name is...'. "
            "Never use a username, handle, or any other name-like string found inside a "
            "URL or email address elsewhere in the background as the sign-off name."
        ),
        expected_output=(
            "A complete cover letter draft with a real salutation and sign-off (no "
            "bracketed placeholders anywhere), ready for the candidate to review."
        ),
        agent=cover_letter_writer,
        context=[analyze_posting, tailor_match],
    )

    fact_check_task = Task(
        description=(
            f"Review the Resume Tailor's confirmed matches and the drafted cover letter "
            f"above for factual accuracy. For EACH specific factual claim (employer "
            f"names, job titles, dates, years of experience, skills, methodologies like "
            f"Agile, certifications), check whether the candidate's real background text "
            f"below actually supports it — not whether it sounds plausible, whether it "
            f"would be a reasonable thing for someone in this field to have, or whether "
            f"a typical resume in this space would include it. Only the literal text "
            f"below counts as evidence.\n\n"
            f"Pay special attention to:\n"
            f"1. Any claim that the candidate currently or previously WORKED AT or WAS "
            f"EMPLOYED BY the company named in the job posting — this is almost always "
            f"wrong, since that company is who they're applying TO, not a past employer. "
            f"IMPORTANT: expressing interest in the role or company ('I'm excited about "
            f"the opportunity at X', 'I'd love to join X') is completely normal and "
            f"correct — do NOT flag simple expressions of interest in applying. Only "
            f"flag it if the text claims actual past/current employment there.\n"
            f"2. Any job title, skill, or methodology (e.g. 'lead analyst', 'Agile') "
            f"that isn't an exact or near-exact match to something stated below. Do not "
            f"invent or paraphrase supporting evidence that isn't literally present in "
            f"the background text — if you're unsure whether something is supported, "
            f"you must be able to quote the exact supporting sentence from the "
            f"background below. If you cannot produce that exact quote, flag it.\n"
            f"3. Any years-of-experience figure that doesn't match a figure the "
            f"candidate's own background explicitly states about themselves.\n\n"
            f"List every claim you cannot verify in flagged_claims, each as a short "
            f"quote of the claim plus a brief note on why it's unverified. flagged_claims "
            f"must contain ONLY genuinely unverifiable claims — do not include any note, "
            f"commentary, or justification about claims that ARE verified; simply leave "
            f"those out of the list entirely. Set verified=True only if you find zero "
            f"issues.\n\n"
            f"Candidate's real background (the only source of truth):\n"
            f"{candidate_background_text}"
        ),
        expected_output=(
            "A list of specific unverifiable claims (or an empty list if none), and a "
            "verified boolean."
        ),
        agent=fact_checker,
        context=[tailor_match, write_letter],
        output_pydantic=FactCheckResult,
    )

    crew = Crew(
        agents=[job_analyst, resume_tailor, ats_formatter, cover_letter_writer, fact_checker],
        tasks=[analyze_posting, tailor_match, ats_formatting_task, write_letter, fact_check_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew, {
        "analyze_posting": analyze_posting,
        "tailor_match": tailor_match,
        "ats_formatting_task": ats_formatting_task,
        "write_letter": write_letter,
        "fact_check_task": fact_check_task,
    }


def run_deterministic_checks(cover_letter_text: str, company_name: str) -> list:
    issues = []
    text_lower = cover_letter_text.lower()

    true_years_match = re.search(r"(\d+)\+?", YEARS_OF_EXPERIENCE_SUMMARY)
    if true_years_match:
        true_years = true_years_match.group(1)
        for found_years in re.findall(r"(\d+)\+?\s*years?", text_lower):
            if found_years != true_years:
                issues.append(
                    f"[Automated check] Letter states '{found_years} years' of "
                    f"experience, but the candidate's real figure is "
                    f"'{YEARS_OF_EXPERIENCE_SUMMARY}' — likely echoing the job "
                    f"posting's required years instead of the candidate's own."
                )

    if company_name and company_name.strip().lower() not in ("not specified", ""):
        company_lower = company_name.strip().lower()
        employment_indicator_pattern = re.compile(
            r"(currently|since \d{4}|employed (at|by)|working at|work at|"
            r"as a .{0,40} at)\s+.{0,30}" + re.escape(company_lower)
        )
        if employment_indicator_pattern.search(text_lower):
            issues.append(
                f"[Automated check] Letter contains language suggesting current or "
                f"past employment at '{company_name}' — this is the company being "
                f"applied to, not a real employer, unless explicitly confirmed "
                f"elsewhere in the candidate's background."
            )

    return issues


def check_ats_skills_grounding(skills: list, candidate_background_text: str) -> list:
    issues = []
    background_lower = candidate_background_text.lower()

    for skill in skills:
        skill_clean = skill.strip()
        if skill_clean and skill_clean.lower() not in background_lower:
            issues.append(
                f"[Automated check] Skill '{skill_clean}' doesn't appear anywhere in "
                f"the candidate's real background text — worth checking whether this "
                f"came from the job posting's own wording rather than the candidate's "
                f"actual background."
            )

    return issues


if __name__ == "__main__":
    print("Paste the job posting, then press Ctrl+D (or Ctrl+Z on Windows) when done:", file=sys.stderr)
    job_posting = sys.stdin.read()
    if not job_posting.strip():
        print("No job posting text provided.", file=sys.stderr)
        sys.exit(1)

    try:
        candidate_background = load_candidate_background()
        print(f"Loaded background from ChromaDB ({len(candidate_background)} characters).", file=sys.stderr)
    except Exception as e:
        print(f"Couldn't load background from ChromaDB: {e}", file=sys.stderr)
        print("Falling back to a minimal hardcoded summary — results may be less specific.", file=sys.stderr)
        candidate_background = f"My name is {CANDIDATE_NAME}. {CURRENT_ROLE_SUMMARY}"

    crew, tasks = build_crew(job_posting, candidate_background)
    crew.kickoff()

    final_text = getattr(tasks["write_letter"].output, "raw", None) or ""
    with open(RESULT_OUTPUT_FILE, "w") as f:
        f.write(final_text)

    structured_outputs = [
        (tasks["analyze_posting"], JOB_ANALYSIS_OUTPUT_FILE, "job analysis"),
        (tasks["tailor_match"], MATCH_ANALYSIS_OUTPUT_FILE, "match analysis"),
        (tasks["fact_check_task"], FACT_CHECK_OUTPUT_FILE, "fact check"),
    ]
    for task_obj, output_path, label in structured_outputs:
        try:
            task_output = task_obj.output
            if task_output is not None and task_output.pydantic is not None:
                with open(output_path, "w") as f:
                    json.dump(task_output.pydantic.model_dump(), f, indent=2)
                print(f"{label.capitalize()} data written to {output_path}", file=sys.stderr)
            else:
                print(
                    f"Warning: {label} task did not return structured (pydantic) "
                    f"output — check your installed CrewAI version supports output_pydantic.",
                    file=sys.stderr,
                )
        except Exception as e:
            print(f"Warning: failed to write {label} output: {e}", file=sys.stderr)

    ats_sections = None
    try:
        ats_task_output = tasks["ats_formatting_task"].output
        if ats_task_output is not None and ats_task_output.raw:
            ats_sections = parse_ats_text_to_sections(ats_task_output.raw)
            with open(ATS_RESULT_OUTPUT_FILE, "w") as f:
                json.dump(ats_sections, f, indent=2)
            print(f"ATS resume data written to {ATS_RESULT_OUTPUT_FILE}", file=sys.stderr)
        else:
            print("Warning: ATS formatter task produced no output to parse.", file=sys.stderr)
    except Exception as e:
        print(f"Warning: failed to parse/write ATS resume output: {e}", file=sys.stderr)

    try:
        company_name_for_check = ""
        job_analysis_output = tasks["analyze_posting"].output
        if job_analysis_output is not None and job_analysis_output.pydantic is not None:
            company_name_for_check = job_analysis_output.pydantic.company_name

        deterministic_issues = run_deterministic_checks(final_text, company_name_for_check)

        if ats_sections is not None:
            deterministic_issues += check_ats_skills_grounding(
                ats_sections.get("skills", []), candidate_background_text
            )

        if os.path.isfile(FACT_CHECK_OUTPUT_FILE):
            with open(FACT_CHECK_OUTPUT_FILE, "r") as f:
                fact_check_data = json.load(f)
        else:
            fact_check_data = {"flagged_claims": [], "verified": True}

        if deterministic_issues:
            fact_check_data["flagged_claims"] = fact_check_data.get("flagged_claims", []) + deterministic_issues
            fact_check_data["verified"] = False

        with open(FACT_CHECK_OUTPUT_FILE, "w") as f:
            json.dump(fact_check_data, f, indent=2)

        if deterministic_issues:
            print(f"Deterministic checks found {len(deterministic_issues)} issue(s).", file=sys.stderr)
        else:
            print("Deterministic checks found no issues.", file=sys.stderr)
    except Exception as e:
        print(f"Warning: deterministic checks failed to run: {e}", file=sys.stderr)

    print("===FINAL_RESULT_WRITTEN===", file=sys.stderr)
    print(f"Clean result written to {RESULT_OUTPUT_FILE}", file=sys.stderr)
