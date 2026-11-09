# Opportunity Search Tool

A local web app that finds U.S. federal business opportunities for a small company and
scores each one against that company's capabilities. It covers four kinds of opportunity:

| Type | What it finds | Where it looks |
|---|---|---|
| **Contracts** | Solicitations, presolicitations, sources-sought notices | SAM.gov, deep web search |
| **Small business** | Set-aside procurements the company is eligible for, forecasts, subcontracting | SAM.gov, deep web search |
| **SBIR/STTR** | Open and upcoming topics and funding notices | DoD SBIR/STTR portal, Grants.gov, SAM.gov, deep web search |
| **CRADA** | Agencies and labs seeking research partners | Federal Register, SAM.gov, deep web search, built-in agency directory |

Each search shows live progress. Results are ranked by fit and winnability, with a short
explanation, and can be moved through a capture pipeline from New to Won or Lost.

## Before you rely on it

This is an early version. Please read these four points first.

1. **Test the SAM.gov connection first.** SAM.gov supplies most contract and small-business results.
   The connection was built from SAM.gov's official API documentation and passes offline tests, but
   it has not yet been run against the live service. Add your key, run one search, and check the
   activity feed for a `SAM.gov: scanned N → M match` line. If it shows an error, please report it.
2. **SAM.gov limits may be low.** Daily request limits depend on your account's role, so each run
   makes as few calls as possible. Full notice text is off by default, so SAM.gov listings are
   scored from title, agency, NAICS, and set-aside, and capped at 60. If your key has a higher limit,
   set `SAM_MAX_DESCRIPTIONS` in `.env` (for example, `50`).
3. **SBIR.gov coverage is indirect.** SBIR.gov's own API refused automated access when this was
   built, so civilian SBIR/STTR coverage comes from Grants.gov, SAM.gov, and deep web search.
4. **No license has been granted yet.** This repository does not include a license. Please contact
   the author about terms before using it in production or redistributing it.

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/sprawl-Rob/opportunity-search-tool.git
cd opportunity-search-tool
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and add:

1. **`SAM_API_KEY`** — free. Sign in at [sam.gov](https://sam.gov), open **Account Details**, and
   request a public API key. Without it, contracts and small-business set-asides won't appear.
2. **At least one AI key** — `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY`. Each run
   uses your own account and is billed to it. Google Search grounding requires billing to be
   enabled on the Google key.

Then start the app and open <http://127.0.0.1:8010>:

```bash
python app.py
```

## First run

1. Open **⚙ Company profile** and fill in your company name, capabilities, NAICS codes, keywords,
   certifications, and past performance. The better this is, the better the scoring.
2. Choose the AI models, or keep the defaults. **↻ Refresh models** gets each provider's current list.
3. Click **Run search**.

**Certifications matter.** Set-asides that need a certification you don't list (8(a), HUBZone,
SDVOSB, WOSB, EDWOSB) are hidden. Use one per line: `8a`, `hubzone`, `sdvosb`, `vosb`, `wosb`,
`edwosb`. The app assumes the company qualifies as small under the relevant NAICS code; confirm
that for each pursuit.

## Things to know

- **A CRADA is not a contract.** Federal labs generally cannot pay the partner. CRADAs are scored on
  strategic value: lab access, IP, credibility, and a path to funded work.
- **Always confirm on the official notice.** Deadlines, amendments, and eligibility change. Listings
  marked *link unverified* could not be confirmed automatically (many `.mil` sites and SAM.gov links
  can't be checked by a program).
- **Scoring costs are limited.** An opportunity is rescored only when it's new or when the company
  profile or scoring model changes.

## Your data

Everything stays on your computer in `opportunities.db`, `profile.json`, `settings.json`, and `.env`.
The app listens only on `127.0.0.1`. Company profile text is sent to the AI provider you choose so
it can search and score, so don't put controlled or classified information in the profile.

## Tests

```bash
python tests/test_sources.py
```
