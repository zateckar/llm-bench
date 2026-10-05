"""Built-in open-ended assistant suite (``assistant-open-v1``).

Thirty everyday workplace requests in English, Czech and German. They have no
answer key: runs record the answers (outcome ``recorded``) and blind A/B
studies compare them pairwise, by people and LLM judges. The criteria tell
both what a good answer must do; a reference is given only where one exists.
"""

from __future__ import annotations

from app.benchmarking.models import Question
from app.benchmarking.quality_suite import MAX_OUTPUT_TOKENS

REVISION = "assistant-open-v1"
LANGUAGES = ("en", "cs", "de")
SYSTEM = (
    "You are the internal AI assistant of a large car manufacturer. Help employees with "
    "everyday work. Answer in the language of the request unless asked otherwise. Be accurate, "
    "concise and practical; when information is missing, say so instead of inventing it."
)

# (category, language, prompt, criteria, reference)
ITEMS = [
    # --- Writing ---------------------------------------------------------------
    ("Writing", "en",
     "Write an email to our supplier contact, Mark Fischer, postponing Thursday's review meeting "
     "about the delayed battery modules. Our quality lead is ill. Propose two new dates next week "
     "and ask him to send an updated delivery forecast before the meeting.",
     ["Professional, polite tone suitable for an external supplier",
      "States the postponement and gives the reason briefly without oversharing",
      "Proposes two concrete alternative dates next week",
      "Asks for an updated delivery forecast before the meeting",
      "Concise: no more than about 150 words"], None),
    ("Writing", "cs",
     "Napište krátké interní oznámení pro zaměstnance: v sobotu 14. listopadu od 8:00 do 14:00 "
     "proběhne údržba IT. SAP a docházkový systém nebudou dostupné, e-mail a Teams budou fungovat. "
     "Případné problémy hlaste na helpdesk, linka 4444.",
     ["Written in Czech", "States the exact date and time window",
      "Says which systems are unavailable and which keep working",
      "Tells employees what to do, including the helpdesk line 4444",
      "Short and easy to scan"], None),
    ("Writing", "de",
     "Schreiben Sie eine freundliche Absage an Frau Keller, die sich als Qualitätsingenieurin bei "
     "uns beworben hat. Wir haben uns für eine Kandidatin mit mehr Erfahrung in der Lieferantenentwicklung "
     "entschieden. Ihr Profil möchten wir gern für künftige Stellen behalten, wenn sie zustimmt.",
     ["Written in German with appropriate formal address (Sie)",
      "Thanks the applicant and states the decision clearly",
      "Gives no legally risky or discriminatory reasons",
      "Asks for consent to keep the profile instead of assuming it",
      "Warm but makes no promises"], None),
    ("Writing", "en",
     "Draft an agenda for a 60-minute kickoff meeting for a project that moves our plant "
     "maintenance tickets from spreadsheets to a ticketing system. Attendees: maintenance "
     "supervisors, IT, and the plant manager.",
     ["Covers goals, scope, roles, timeline, risks and next steps",
      "Time allocations add up to roughly 60 minutes",
      "Reflects the different attendees' interests",
      "Ends with clear decisions or action items to capture"], None),
    ("Writing", "cs",
     "Napište žádost svému vedoucímu o schválení třídenního školení Kubernetes v Praze za 18 000 Kč. "
     "Pracuji jako DevOps inženýr a náš tým příští rok přechází na kontejnerovou platformu.",
     ["Written in Czech with a polite but direct tone",
      "States the course, duration, place and cost",
      "Explains the benefit for the team's planned container migration",
      "Asks clearly for approval"], None),
    # --- Summarising ------------------------------------------------------------
    ("Summarising", "en",
     "Summarise these meeting notes as decisions, action items with owners and dates, and open "
     "questions.\n\nNotes: Weekly logistics sync. Anna said the new forklift routes cut transport time "
     "by about 10% in hall 4. We agreed to roll the routes out to hall 5 from 1 December. Tomas will "
     "update the floor markings by 28 November. Pallet damage is still high at gate B; nobody knows "
     "if it is the supplier's packaging or our unloading. Petra will collect photos for two weeks. "
     "The budget for new scanners was discussed but not decided; finance needs a quote first. "
     "Jakub will ask two vendors for quotes by the end of the month.",
     ["Lists the decision to roll out routes to hall 5 from 1 December",
      "Lists action items with the right owners and deadlines (Tomas, Petra, Jakub)",
      "Captures open questions: the cause of pallet damage and the scanner budget",
      "Adds no facts that are not in the notes"], None),
    ("Summarising", "cs",
     "Shrňte tento e-mail zákazníka do tří bodů pro kolegu z reklamací.\n\n„Dobrý den, 3. října jsem "
     "si u vašeho dealera v Brně vyzvedl nový vůz. Po týdnu se začala rozsvěcovat kontrolka tlaku v "
     "pneumatikách, i když tlak je v pořádku – nechal jsem ho dvakrát zkontrolovat. Dealer mi řekl, "
     "že to je běžné a mám počkat. Kontrolka svítí dál a navigace občas ztrácí signál GPS. Chtěl bych "
     "termín v servisu do konce měsíce a vyjádření, zda jde o záruční vadu. S pozdravem, Jan Dvořák“",
     ["Written in Czech as exactly three bullet points",
      "Mentions the tyre-pressure warning despite correct pressure and the GPS signal loss",
      "Mentions the dealer's response and the customer's requests (service date, warranty)",
      "Adds no invented details"], None),
    ("Summarising", "de",
     "Fassen Sie diesen Bericht in höchstens fünf Sätzen für die Werksleitung zusammen.\n\n"
     "Im dritten Quartal stieg die Ausschussquote in der Lackiererei von 2,1 % auf 3,4 %. Die "
     "Analyse zeigt, dass 70 % der zusätzlichen Fehler Staubeinschlüsse sind. Ursache ist ein "
     "verspäteter Filterwechsel in Kabine 3, weil Ersatzfilter zwei Wochen lang nicht lieferbar "
     "waren. Seit dem Wechsel am 12. September liegt die Quote wieder bei 2,3 %. Die Kosten des "
     "zusätzlichen Ausschusses betragen rund 410 000 Euro. Vorgeschlagen werden ein Sicherheitsbestand "
     "von Filtern für vier Wochen und ein zweiter Lieferant.",
     ["Written in German, at most five sentences",
      "Includes the scrap-rate rise (2.1% to 3.4%), the dust cause and the filter shortage",
      "States that the rate recovered after 12 September and the cost of about 410,000 euros",
      "Mentions both proposed measures",
      "Suitable for management: no unnecessary detail"], None),
    ("Summarising", "en",
     "Give our executives a two- or three-sentence summary of this incident.\n\nOn 2 October from "
     "06:10 to 07:45 the parts-ordering portal was unavailable for suppliers. A certificate on the "
     "load balancer expired because the renewal reminder went to a mailbox of a colleague who left "
     "the company. 140 suppliers could not confirm orders; no orders were lost because the system "
     "queued them. The certificate was renewed, reminders now go to a team mailbox, and automatic "
     "renewal will be enabled by the end of October.",
     ["Two or three sentences only",
      "States impact: 95 minutes of supplier portal outage, 140 suppliers, no lost orders",
      "States the cause (expired certificate, reminder to a departed employee's mailbox)",
      "States the fixes and the planned automatic renewal"], None),
    ("Summarising", "de",
     "Fassen Sie die neue Reisekostenrichtlinie in Stichpunkten für Mitarbeitende zusammen.\n\n"
     "Ab 1. Januar sind Dienstreisen bis 500 km grundsätzlich mit der Bahn anzutreten; Ausnahmen "
     "genehmigt die Abteilungsleitung. Hotelkosten werden bis 130 Euro pro Nacht erstattet, in "
     "München, Paris und London bis 180 Euro. Belege sind innerhalb von 30 Tagen nach Reiseende im "
     "Reiseportal hochzuladen; danach ist keine Erstattung mehr möglich. Taxifahrten werden nur "
     "erstattet, wenn kein öffentliches Verkehrsmittel zumutbar war. Das Tagegeld bleibt unverändert.",
     ["Written in German as bullet points",
      "Includes the 500 km rail rule and who approves exceptions",
      "Includes hotel limits (130 euros, 180 euros in the three named cities)",
      "Includes the 30-day receipt deadline and its consequence, and the taxi rule",
      "Mentions that the daily allowance is unchanged"], None),
    # --- Explaining -------------------------------------------------------------
    ("Explaining", "en",
     "Explain to a non-technical department head what a language model's \"context window\" is "
     "and why it matters when people upload long documents to our assistant.",
     ["Accurate explanation in plain language with no unexplained jargon",
      "Explains the practical consequence: content beyond the limit is not considered or must be split",
      "Gives practical advice for long documents",
      "Brief enough for a busy manager"], None),
    ("Explaining", "cs",
     "Vysvětlete týmu, který začíná s agilním řízením, rozdíl mezi Scrumem a Kanbanem a kdy se "
     "který hodí.",
     ["Written in Czech", "Correctly describes Scrum (sprints, roles, ceremonies)",
      "Correctly describes Kanban (continuous flow, WIP limits, board)",
      "Gives concrete guidance on when each fits",
      "Clear structure, not overly long"], None),
    ("Explaining", "de",
     "Erklären Sie einem neuen Einkäufer, was der PPAP (Production Part Approval Process) ist und "
     "warum er für uns wichtig ist.",
     ["Written in German", "Correctly explains PPAP's purpose and when it is required",
      "Names typical elements (e.g. control plan, PFMEA, measurement results, PSW)",
      "Explains why it matters for supply quality and the buyer's role"], None),
    ("Explaining", "en",
     "Why does this Python code print [1, 2] on the second call instead of [2]?\n\n"
     "```python\ndef add(item, items=[]):\n    items.append(item)\n    return items\n\n"
     "add(1)\nprint(add(2))\n```",
     ["Correctly identifies the mutable default argument evaluated once at definition",
      "Shows the idiomatic fix with None as the default",
      "Explanation is concise and correct"],
     "The default list is created once when the function is defined, so every call without "
     "`items` shares it. Use `items=None` and create a new list inside the function."),
    ("Explaining", "cs",
     "Vysvětlete novému kolegovi z kvality, co je 8D report, z jakých kroků se skládá a kdy se "
     "používá.",
     ["Written in Czech", "Lists the eight disciplines correctly and in order",
      "Explains when 8D is used (customer complaints, recurring or serious defects)",
      "Practical and well structured"], None),
    # --- Advising ---------------------------------------------------------------
    ("Advising", "en",
     "My team of six engineers spends about 15 hours a week each in meetings and complains they "
     "cannot get work done. What should I change? Give me a concrete plan.",
     ["Gives specific, actionable steps rather than generic advice",
      "Prioritises and suggests how to measure improvement",
      "Considers the team's input and which meetings are essential",
      "Realistic for a team lead to implement"], None),
    ("Advising", "cs",
     "Za dva týdny mám jako vedoucí roční hodnoticí rozhovor s podřízeným, jehož výkon se letos "
     "zhoršil. Jak se na rozhovor připravit a jak ho vést?",
     ["Written in Czech",
      "Advises preparing concrete examples and facts rather than impressions",
      "Recommends listening and exploring causes, including possible personal reasons",
      "Ends with agreed goals, support and follow-up",
      "Respectful toward the employee"], None),
    ("Advising", "de",
     "Wir müssen zwischen zwei Lieferanten für Kabelbäume wählen. A ist 8 % günstiger, hat aber "
     "sechs Wochen Lieferzeit und keine Erfahrung mit uns. B ist teurer, liefert in zwei Wochen und "
     "hatte im letzten Jahr zwei Qualitätsprobleme. Wie sollten wir entscheiden?",
     ["Written in German",
      "Proposes structured criteria (total cost, risk, quality, capacity, dependency)",
      "Discusses the trade-offs of both suppliers using the given facts",
      "Suggests what information to obtain or how to reduce risk (audit, trial order, dual sourcing)",
      "Does not invent facts about the suppliers"], None),
    ("Advising", "en",
     "A colleague asked me to share my password so he can finish a report in our system while I "
     "am on holiday next week. What should I do and what should I reply?",
     ["Clearly advises not to share the password",
      "Offers legitimate alternatives (delegation, access request via IT, sharing the file)",
      "Provides a short, friendly reply the user can send"], None),
    ("Advising", "de",
     "Unsere Abteilung erstellt monatlich 20 Excel-Berichte manuell. Wir wollen auf Power BI "
     "umsteigen. Wie sollten wir vorgehen?",
     ["Written in German",
      "Proposes phases (inventory, data sources, pilot, rollout, training)",
      "Mentions data quality, ownership and access rights",
      "Recommends starting with a pilot on a few reports",
      "Practical and not vendor marketing"], None),
    # --- Transforming data ------------------------------------------------------
    ("Transforming data", "en",
     "Convert this CSV into a Markdown table sorted by cost, highest first.\n\n"
     "part,supplier,cost_eur\nbrake disc,Brembo,48.20\nwiper blade,Bosch,7.90\n"
     "headlamp,Hella,312.00\nair filter,Mann,12.40",
     ["Valid Markdown table with the three columns",
      "Rows sorted by cost descending: headlamp, brake disc, air filter, wiper blade",
      "Values copied exactly",
      "No unnecessary commentary"],
     "| part | supplier | cost_eur |\n|---|---|---|\n| headlamp | Hella | 312.00 |\n"
     "| brake disc | Brembo | 48.20 |\n| air filter | Mann | 12.40 |\n| wiper blade | Bosch | 7.90 |"),
    ("Transforming data", "cs",
     "Z následujícího textu vytvořte JSON pole objektů s klíči jmeno, oddeleni a telefon.\n\n"
     "Na poradě byli Petra Malá z nákupu (tel. 326 811 204), za logistiku Ondřej Černý, linka "
     "326 811 377, a Lucie Horáková z kvality, která telefon neuvedla.",
     ["Valid JSON array with three objects and the requested keys",
      "Names, departments and numbers are correct",
      "Missing phone is null or empty, not invented"],
     '[{"jmeno": "Petra Malá", "oddeleni": "nákup", "telefon": "326 811 204"}, '
     '{"jmeno": "Ondřej Černý", "oddeleni": "logistika", "telefon": "326 811 377"}, '
     '{"jmeno": "Lucie Horáková", "oddeleni": "kvalita", "telefon": null}]'),
    ("Transforming data", "de",
     "Übersetzen Sie diese E-Mail ins Englische und behalten Sie den Ton bei.\n\n"
     "„Hallo Herr Novák, vielen Dank für die schnelle Lieferung der Muster. Leider haben zwei der "
     "zehn Teile Grate an der Bohrung. Könnten Sie bis Freitag eine kurze Stellungnahme schicken? "
     "Die restlichen Teile sind einwandfrei. Beste Grüße, Sabine Wolf“",
     ["Complete, accurate English translation",
      "Keeps the polite, friendly business tone",
      "Correctly renders technical detail (burrs on the bore, two of ten parts)",
      "No added or omitted content"],
     "Hello Mr Novák, thank you for the quick delivery of the samples. Unfortunately, two of the "
     "ten parts have burrs on the bore. Could you send a short statement by Friday? The remaining "
     "parts are flawless. Best regards, Sabine Wolf"),
    ("Transforming data", "en",
     "Turn these notes into a bug report with title, steps to reproduce, expected result, actual "
     "result and environment.\n\nnotes: shift planner app, Chrome 128 on win 11. when you copy a "
     "week to next week and the week has a holiday the night shift on the holiday disappears. should "
     "copy normally and just warn about holiday. happens every time, tried with week 44.",
     ["Contains all requested sections",
      "Steps are specific and reproducible from the notes",
      "Expected and actual results are correct and distinct",
      "Includes the environment (Chrome 128, Windows 11) and adds no invented details"], None),
    ("Transforming data", "cs",
     "Přeložte do češtiny:\n\n\"The torque wrench must be calibrated every six months or after "
     "5,000 cycles, whichever comes first. Record the calibration date on the label and in the "
     "maintenance system.\"",
     ["Accurate, natural Czech translation",
      "Correct technical terms (momentový klíč, kalibrace, cykly)",
      "Preserves the 'whichever comes first' condition",
      "No additions"],
     "Momentový klíč musí být kalibrován každých šest měsíců nebo po 5 000 cyklech, podle toho, "
     "co nastane dříve. Datum kalibrace zapište na štítek a do systému údržby."),
    # --- Reviewing --------------------------------------------------------------
    ("Reviewing", "en",
     "Review this function and fix any bugs.\n\n```python\ndef average_delay(delays):\n"
     "    total = 0\n    for i in range(1, len(delays)):\n        total += delays[i]\n"
     "    return total / len(delays)\n```",
     ["Finds the skipped first element (range starts at 1)",
      "Finds the division by zero for an empty list",
      "Provides a corrected version that handles both",
      "Explanation is concise"], None),
    ("Reviewing", "cs",
     "Zkontrolujte tento e-mail zákazníkovi před odesláním a navrhněte opravenou verzi.\n\n"
     "„Dobrý den, k vaší reklamaci vám sdělujem, že vada nebyla potvrzena a proto ji nemůžeme "
     "uznat. Pokud nesouhlasíte, je to vaše věc, můžete se obrátit jinam. S pozdravem“",
     ["Corrects the grammar error (sdělujeme)",
      "Points out the dismissive, unprofessional tone",
      "Proposed version is polite, explains the next steps or the appeal option",
      "Written in Czech"], None),
    ("Reviewing", "de",
     "Prüfen Sie diesen Code auf Probleme.\n\n```python\ndef find_orders(cursor, customer):\n"
     "    query = \"SELECT * FROM orders WHERE customer = '\" + customer + \"'\"\n"
     "    cursor.execute(query)\n    return cursor.fetchall()\n```",
     ["Written in German",
      "Identifies the SQL injection risk from string concatenation",
      "Shows a parameterised query as the fix",
      "Optionally mentions SELECT * or unbounded results as minor points"], None),
    ("Reviewing", "en",
     "Review this requirement for ambiguity and rewrite it to be testable: \"The system should "
     "respond quickly and handle a large number of users without problems.\"",
     ["Identifies vague terms (quickly, large number, without problems)",
      "Rewrites with measurable targets (e.g. latency percentile, concurrent users, error rate)",
      "Notes that the targets are placeholders to agree with stakeholders"], None),
    ("Reviewing", "cs",
     "Proveďte code review tohoto JavaScriptu.\n\n```javascript\nasync function loadUsers(ids) {\n"
     "  const users = [];\n  ids.forEach(async (id) => {\n    const res = await fetch(`/api/users/${id}`);\n"
     "    users.push(await res.json());\n  });\n  return users;\n}\n```",
     ["Written in Czech",
      "Identifies that forEach does not await, so an empty array is returned",
      "Suggests Promise.all with map (or a for...of loop)",
      "Mentions missing error handling for failed responses"], None),
]

SLUGS = {"Writing": "writing", "Summarising": "summarising", "Explaining": "explaining",
         "Advising": "advising", "Transforming data": "transforming", "Reviewing": "reviewing"}


def load_questions():
    questions, counters = [], {}
    for category, lang, prompt, criteria, reference in ITEMS:
        counters[category] = counters.get(category, 0) + 1
        ident = f"open-{SLUGS[category]}-{counters[category]:02d}-{lang}"
        expected = {"criteria": list(criteria)}
        if reference is not None:
            expected["reference"] = reference
        questions.append(Question(
            id=ident, category=category, prompt=prompt, evaluator="open_ended", expected=expected,
            system_prompt=SYSTEM, max_tokens=MAX_OUTPUT_TOKENS,
            metadata={"family": ident, "lang": lang, "scope": "open_ended", "cohort": REVISION},
        ))
    if len({q.id for q in questions}) != len(questions):
        raise ValueError("Duplicate question IDs in the assistant-open suite")
    return questions


def provenance():
    return {"revision": REVISION, "questions": len(ITEMS), "languages": list(LANGUAGES),
            "categories": list(SLUGS)}
