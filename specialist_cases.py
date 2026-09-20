"""Deterministic authoring for specialist-v1. Regenerate with this module.

Only prompts go to models. Answer derivations and review rationales stay here.
Legal rules and market inputs are bounded exercise assumptions, not live advice.
"""

from fractions import Fraction
from pathlib import Path

import yaml

from review_cases import review_cases
from translation_cases import translation_cases


def make_case(id, category, prompt, answer, rationale, source="Original synthetic exercise"):
    return {
        "id": id,
        "category": category,
        "difficulty": "expert",
        "description": rationale,
        "source": source,
        "prompt": prompt.strip()
        + "\nReturn only one JSON document with exactly the requested keys. "
        "Arrays must be in the specified order. Do not add explanations or extra claims.",
        "evaluator": "json_match",
        "expected": {"value": answer, "mode": "exact", "strict_json": True},
        "max_tokens": 16384,
    }


def build_cases():
    cases = []

    def add(*args, **kwargs):
        cases.append(make_case(*args, **kwargs))

    add(
        "SP1-LE-01",
        "Legal",
        """
European patent examination exercise; apply ONLY this stipulated rule pack.
Each claim has a valid effective date stated below. Novelty is lost only if ONE
eligible document directly discloses ALL claim elements together; do not mosaic.
Public documents before the effective date are eligible for novelty and inventive
step. An earlier-filed, later-published EP application meeting Article 54(3) is
eligible for novelty only. Here all such formal conditions are stipulated met.
Dates and element sets:
C1: effective 2025-02-01, {A,B}; C2: effective 2025-06-01, {A,B,C}.
D1: public 2025-01-10, {A}; D2: public 2025-01-20, {B,C};
D3: EP filed 2025-01-05, published 2025-08-01, {A,B};
D4: public 2025-04-01, {A,B,C}; D5: public 2025-07-01, {A,B,C,D}.
For each claim return {novelty_destroyers:[document IDs],
inventive_step_eligible:[document IDs],inventive_step_decidable:bool}.
The last field asks whether an inventive-step conclusion can be determined from
element sets alone without evidence about skilled-person reasoning; novelty and
inventive step must be assessed separately. Sort IDs. Top keys C1,C2.
""",
        {
            "C1": {
                "novelty_destroyers": ["D3"],
                "inventive_step_eligible": ["D1", "D2"],
                "inventive_step_decidable": False,
            },
            "C2": {
                "novelty_destroyers": ["D4"],
                "inventive_step_eligible": ["D1", "D2", "D4"],
                "inventive_step_decidable": False,
            },
        },
        "Distinguishes single-document novelty, effective dates, and novelty-only prior art; element overlap does not establish obviousness.",
        "EPC Article 56: https://www.epo.org/en/legal/epc/2020/a56.html ; stipulated Article 54 eligibility",
    )

    add(
        "SP1-LE-02",
        "Legal",
        """
Patent claim-chart/FTO exercise. Apply the following simplified literal-scope rules,
not unstated doctrines of equivalents. An active granted patent blocks an unlicensed
product in its territory iff every element of at least one claim is present.
An expired patent blocks nothing; pending claims do not establish a current block.
A product's own patent does not establish clearance against other patents.
Products: X={motor,sensor,controller}; Y={motor,controller}; Z={motor,sensor,optical}.
P1 active DE, claims {motor,sensor}; P2 active CZ, claims {motor,controller};
P3 expired DE, claims {motor}; P4 pending DE, claims {motor,controller};
P5 active DE, owned by X's maker, claims {motor,sensor,controller}.
X's maker licenses P1 for Germany only. Y and Z have no licenses. P5 is licensed
to neither Y nor Z. For each product list confirmed blocking IDs in DE and CZ.
Return {X:{DE:[],CZ:[]},Y:{DE:[],CZ:[]},Z:{DE:[],CZ:[]},
X_worldwide_clearance_established:bool,P4_current_block_established:bool}.
ID lists sorted. No facts about other jurisdictions or patents are supplied.
""",
        {
            "X": {"DE": [], "CZ": ["P2"]},
            "Y": {"DE": [], "CZ": ["P2"]},
            "Z": {"DE": ["P1"], "CZ": []},
            "X_worldwide_clearance_established": False,
            "P4_current_block_established": False,
        },
        "Element-by-element territorial chart; licensed P1 and owned P5 do not clear Czech P2 or unknown worldwide rights.",
        "WIPO patent FAQ: https://www.wipo.int/en/web/patents/faq_patents ; simplified literal-scope exercise",
    )

    add(
        "SP1-LE-03",
        "Legal",
        """
US trademark screening exercise, evidence closed as of 2026-01-15.
Rules for this exercise: similarity of marks AND related goods/services create a
screening conflict, not a final adjudication. Nice class equality alone does not
prove relatedness; different classes do not rule it out. No search result does not
prove clearance. Stipulated evidence supplies similarity and relatedness separately.
Applicant NOVARA, vehicle diagnostic software, class 9. Search results:
T1 NOVARRA diagnostic SaaS class 42: similar sound; goods/services related by evidence.
T2 NOVARA industrial abrasive wheels class 7: identical mark; relatedness unknown.
T3 LUMEN desktop themes class 9: marks stipulated dissimilar; relatedness unknown.
T4 NOVARA downloadable vehicle diagnostic tools class 9: identical, related.
For T1..T4 classify as conflict, no_conflict_under_rules, or unresolved.
Also classify these assertions as supported, refuted, or unresolved:
A different classes exclude T1; B registration is guaranteed if T1 and T4 are removed;
C identical spelling by itself resolves T2 as a conflict;
D this screen establishes whether any unregistered user has priority.
Return {screen:[four classifications],assertions:[four classifications]}.
For B and D classify the claimed guarantee/establishment, not whether registration
will ultimately occur or whether an unregistered user actually exists.
""",
        {
            "screen": ["conflict", "unresolved", "no_conflict_under_rules", "conflict"],
            "assertions": ["refuted", "refuted", "refuted", "refuted"],
        },
        "Relatedness and similarity are separate; class shortcuts and guarantees exceed the evidence.",
        "USPTO likelihood of confusion: https://www.uspto.gov/trademarks/search/likelihood-confusion ; stipulated screening rule",
    )

    add(
        "SP1-LE-04",
        "Legal",
        """
Fictional negotiated engineering contract. Interpret only these clauses, with no
external law, implied rights, or calendar rules. Business days are Mon-Fri; no holidays.
Ownership of each deliverable transfers only after BOTH full payment and written
acceptance. Embedded background IP remains supplier-owned; customer receives a
perpetual internal-use license on transfer, with no sublicensing right. Confidentiality
lasts 3 years after termination except trade secrets, which last while secret.
Defect notices must arrive within 5 business days AFTER delivery (delivery day excluded).
Termination does not undo ownership or surviving licenses already vested.
Delivery Fri 2026-01-09. Notice Fri 2026-01-16. Acceptance Jan 14; payment Jan 20;
termination Jan 21. The CAD deliverable includes supplier background library B.
On 2030-01-22, technical item S is still a trade secret; ordinary pricing item P is not.
Return {notice_deadline:"YYYY-MM-DD",notice_timely:bool,
owner_on_jan19:"supplier"|"customer",owner_on_jan22:...,B_owner:...,
B_internal_license_survives:bool,B_sublicense_allowed:bool,
confidential_on_2030_01_22:[IDs sorted]}.
""",
        {
            "notice_deadline": "2026-01-16",
            "notice_timely": True,
            "owner_on_jan19": "supplier",
            "owner_on_jan22": "customer",
            "B_owner": "supplier",
            "B_internal_license_survives": True,
            "B_sublicense_allowed": False,
            "confidential_on_2030_01_22": ["S"],
        },
        "Conjunctive vesting, business-day deadline, retained background IP, and survival exceptions.",
    )

    # Independently reproduced by balance simulations and Fraction tests.
    cash, drawn = 70, 0
    flows = [-95, 50, -60, 45]
    ladder = []
    for net in flows:
        cash += net
        draw = max(20 - cash, 0)
        cash += draw
        drawn += draw
        repay = min(max(cash - 20, 0), drawn)
        cash -= repay
        drawn -= repay
        ladder.append({"draw": draw, "repay": repay, "cash": cash, "debt": drawn})
    add(
        "SP1-FI-01",
        "Finances",
        """
Treasury cash ladder, all amounts in thousands EUR, no fees/interest. Opening total
bank cash 100 includes 30 legally restricted throughout; restricted cash cannot be
spent or used to satisfy a 20 unrestricted closing cash floor. Revolver starts at 0,
limit 80; same-day draw and repayment allowed. At each day's close draw the smallest
amount necessary for the floor, then repay as much outstanding revolver as possible
without breaching the floor. Ignore intraday timing. Net settled flows D1..D4:
-95,+50,-60,+45. A separate 40 sale is booked D2 but settles D5 and is NOT in those
flows. Return {days:[{draw,repay,cash,debt},...],peak_debt,limit_breached:bool,
D4_total_bank_cash}. cash means unrestricted cash; include restricted in final total.
""",
        {
            "days": ladder,
            "peak_debt": max(d["debt"] for d in ladder),
            "limit_breached": any(d["debt"] > 80 for d in ladder),
            "D4_total_bank_cash": cash + 30,
        },
        "Settled cash versus booked receipts, restricted balances, and minimal daily revolver use.",
    )

    add(
        "SP1-FI-02",
        "Finances",
        """
Fictional executable FX quotes, not current market prices. Czech treasury receives
EUR 800000 and pays EUR 300000 on the SAME date, and separately owes CZK 5000000.
Net the EUR flows first. Hedge exactly 60% of the resulting EUR exposure with a
deliverable forward at 24.80 CZK per EUR, no fees. Convert the remaining EUR at
scenario spot 24.00 or 26.00 CZK per EUR. All settlement is simultaneous.
Return {net_EUR,forward_action:"buy_EUR"|"sell_EUR",forward_EUR,
unhedged_EUR,net_CZK_at_24,net_CZK_at_26,scenario_spread_CZK,
reference_rate_is_executable:bool}. Last field: does an informational central-bank
reference rate alone guarantee an executable dealing price? Do not round intermediates.
""",
        {
            "net_EUR": 500000,
            "forward_action": "sell_EUR",
            "forward_EUR": 300000,
            "unhedged_EUR": 200000,
            "net_CZK_at_24": 7240000,
            "net_CZK_at_26": 7640000,
            "scenario_spread_CZK": 400000,
            "reference_rate_is_executable": False,
        },
        "Natural netting, hedge direction, quote units, and residual exposure; no forecast is implied.",
        "Synthetic quotes; ECB reference-rate purpose: https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html",
    )

    interest = Fraction(1200000 * 6 * 30 + 800000 * 6 * 15, 100 * 360)
    add(
        "SP1-FI-03",
        "Finances",
        """
Loan interest and covenant exercise. Simple interest at 6% annual ACT/360.
Accrual days are supplied: principal 1200000 for 30 days, then 800000 for 15 days.
No compounding/fees. At reporting time interest is unpaid. Contract defines covenant
debt as principal PLUS accrued interest PLUS lease debt 250000. Eligible cash is
bank cash 210000 LESS restricted cash 60000. Covenant EBITDA is 260000; maximum
(covenant debt - eligible cash)/EBITDA is 3.50 inclusive. No other adjustments.
Return {interest,covenant_debt,eligible_cash,net_debt,ratio:{numerator,denominator},
compliant:bool,extra_unrestricted_cash_needed}. Ratio must be a reduced exact fraction;
last field is the smallest nonnegative cash injection restoring compliance, debt fixed.
""",
        {
            "interest": int(interest),
            "covenant_debt": 1058000,
            "eligible_cash": 150000,
            "net_debt": 908000,
            "ratio": {"numerator": 227, "denominator": 65},
            "compliant": True,
            "extra_unrestricted_cash_needed": 0,
        },
        "Interest is 6000+2000; net leverage 908000/260000 is just below 3.50, despite gross debt exceeding it.",
    )

    add(
        "SP1-FI-04",
        "Finances",
        """
Treasury stress horizon is end of D2. All amounts thousands EUR. Obligations due
by then total 150. Sources: unrestricted bank cash 35; restricted escrow 20 (unavailable);
T-bill market value 80 with 5% sale haircut, sale cash arrives D2; deposit 50 matures
D3 and cannot be broken; undrawn committed line 70, but current covenant caps usable
draws at 25; uncommitted line 40 unavailable in stress. All usable sources independent.
Policy additionally requires 15 unrestricted cash remaining after paying obligations.
Return {usable_by_D2,obligation_shortfall,policy_shortfall,
sources_used:[IDs sorted],largest_obligation_payable_with_floor}.
IDs: bank,escrow,tbill,deposit,committed,uncommitted. Ignore taxes and interest.
""",
        {
            "usable_by_D2": 136,
            "obligation_shortfall": 14,
            "policy_shortfall": 29,
            "sources_used": ["bank", "committed", "tbill"],
            "largest_obligation_payable_with_floor": 121,
        },
        "Haircut 80 to 76; 35+76+25=136. Excludes future maturity and noncommitted capacity.",
    )

    add(
        "SP1-RD-01",
        "R&D",
        """
Automotive energy model, all assumptions exact. Vehicle mass 2000 kg decelerates
from 20 m/s to 10 m/s on level ground. Wheel kinetic-energy loss = m*(v0^2-v1^2)/2.
Regeneration converts at most 60% of that energy to electrical energy, but battery
acceptance is capped at 20 kW for the entire 5 s event. Actual recovered electrical
energy is the smaller limit. Mechanical braking supplies kinetic loss minus wheel
energy diverted to regen (electrical energy divided by 0.60). Neglect all other losses.
Then battery supplies 30 kW traction for 40 s plus 2 kW auxiliaries for 45 s (including
the brake event). Return integer joules {kinetic_loss_J,recovered_J,
friction_brake_J,net_battery_depletion_J} and {regen_limited_by:"efficiency"|"acceptance"}.
Use nearest integer only on friction_brake_J, half up; other outputs exact.
One flat object with those five keys.
""",
        {
            "kinetic_loss_J": 300000,
            "recovered_J": 100000,
            "friction_brake_J": 133333,
            "net_battery_depletion_J": 1190000,
            "regen_limited_by": "acceptance",
        },
        "Battery charge limit differs from wheel regen energy; auxiliary duration includes braking.",
    )

    add(
        "SP1-RD-02",
        "R&D",
        """
Mechanical design worst-case stack, dimensions in integer micrometres.
Housing internal length H=50000 +/-80. Insert A=20000 +/-30, B=29500 +/-50.
Gap at 20 C is H-A-B. At operating temperature the actual housing grows exactly 60,
A grows 40 and B grows 90 (independent of their manufacture errors). A selected shim
s chosen from [0,100,200,300] occupies gap and has zero tolerance/thermal growth.
Requirement: for EVERY allowed tolerance combination at BOTH temperatures,
remaining gap must be in [100,450] inclusive. Do not use root-sum-square/statistics.
Return {cold_unshimmed:[min,max],hot_unshimmed:[min,max],
feasible_shims:[ascending values],all_units_compliant_with_nominal_200:bool}.
""",
        {
            "cold_unshimmed": [340, 660],
            "hot_unshimmed": [270, 590],
            "feasible_shims": [],
            "all_units_compliant_with_nominal_200": False,
        },
        "Cold upper bound requires shim >=210, hot lower bound requires <=170: impossible even continuously.",
    )

    add(
        "SP1-RD-03",
        "R&D",
        """
Robotics transform and stopping-envelope exercise. Column vectors, metres/seconds.
At the image timestamp robot pose in world is translation (2,3), yaw +90 degrees
(counterclockwise). Camera axes align with robot axes; camera origin is (0.2,0)
in robot frame. Detected point is (1,0.5) in camera frame. The latest robot pose
arrives 0.2 s later at (2,3.4), same yaw; use the image-time pose for reconstruction.
At decision time robot speed toward a different stationary obstacle is 2 m/s.
Total delay before braking is 0.2 s sensor age plus 0.1 s actuation latency; during
delay speed is constant. Braking deceleration is 4 m/s^2. Add 0.25 m safety margin.
Available obstacle distance AT decision time is 1.30 m. Treat the age delay as an
additional conservative policy allowance; do not subtract it from distance again.
Return {world_point:[x,y],using_latest_pose_y_error_m,required_clearance_m,
clearance_deficit_m,safe_under_policy:bool}. Exact decimals, no safety certification implied.
""",
        {
            "world_point": [1.5, 4.2],
            "using_latest_pose_y_error_m": 0.4,
            "required_clearance_m": 1.35,
            "clearance_deficit_m": 0.05,
            "safe_under_policy": False,
        },
        "Compose camera offset before world rotation; timestamp mismatch and latency contribute separate errors.",
    )

    add(
        "SP1-RD-04",
        "R&D",
        """
Automotive R&D trial. Four independently randomized vehicles each run the OLD and
NEW controller in counterbalanced order, matched route/weather. Lower Wh/km is better.
Vehicle summaries (old,new): V1(180,174), V2(200,198), V3(160,164), V4(220,210).
Each summary averages 100 repeated laps; laps are not independent randomized units.
Define improvement as old-new. Compute mean improvement across vehicles and the
exact two-sided sign test ignoring ties: under null each of the 2^n sign patterns
is equally likely; count patterns with |positive_count-negative_count| at least
the observed imbalance. Report reduced p fraction. A different unrandomized fleet
shows 12 Wh/km improvement but also changed tyres and weather; its isolated controller
causal effect cannot be identified from those observations.
Return {independent_units,improvements:[V1..V4],mean_improvement,
sign_test_p:{numerator,denominator},significant_at_0_05:bool,
other_fleet_controller_effect_identified:bool}.
""",
        {
            "independent_units": 4,
            "improvements": [6, 2, -4, 10],
            "mean_improvement": 3.5,
            "sign_test_p": {"numerator": 5, "denominator": 8},
            "significant_at_0_05": False,
            "other_fleet_controller_effect_identified": False,
        },
        "Three positive differences out of four yield (1+4+4+1)/16; repeated laps do not raise randomized sample size.",
    )
    cases.extend(translation_cases(make_case))
    cases.extend(review_cases(make_case))
    return cases


if __name__ == "__main__":
    target = Path(__file__).parent / "tests" / "specialists.yaml"
    target.write_text(
        yaml.safe_dump(build_cases(), allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )
    print(f"Wrote {len(build_cases())} questions to {target}")
