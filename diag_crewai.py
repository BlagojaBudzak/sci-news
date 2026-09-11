"""
Diagnostic: run the reviewer + writer + fact-checker path against the
live LLM with a realistic candidate set.
Not part of the pipeline. Safe to delete after debugging.

WARNING: this will call the LLM multiple times and may take several
minutes on CPU. Do not run in parallel with anything else that uses Ollama.
"""
import time
import traceback

# Import config first so CREWAI_TESTING is set before crewai loads.
import config  # noqa: F401

from src.crew_setup import (
    build_review_crew,
    build_write_crew,
    ReviewerSelection,
    WriterOutput,
)

from src.engine import _parse_output

from src.fact_checker import run_fact_check


def _paper(id_, title, abstract, authors, published, source, doi):
    return {
        "id": id_,
        "title": title,
        "abstract": abstract,
        "url": f"https://example.org/{id_}",
        "authors": authors,
        "published": published,
        "source": source,
        "doi": doi,
    }


CANDIDATES = [
    _paper(
        "test.001",
        "Hard-carbon anodes from biomass for sodium-ion batteries",
        "Hard-carbon anodes derived from biomass precursors were tested in "
        "sodium-ion full cells. The materials delivered 280 mAh/g at C/10 and "
        "retained 92% capacity after 500 cycles. Post-mortem analysis suggests "
        "sodium storage occurs via a combination of adsorption and intercalation.",
        ["A. Author", "B. Author"], "2026-09-01", "arXiv", "10.1000/test.001",
    ),
    _paper(
        "test.002",
        "Fluoroethylene carbonate additive for sodium-ion electrolytes",
        "Fluoroethylene carbonate was investigated as an electrolyte additive "
        "in sodium-ion cells. Cells containing 2 wt% FEC exhibited improved "
        "coulombic efficiency and reduced gas evolution versus baseline cells.",
        ["C. Author"], "2026-09-02", "arXiv", "10.1000/test.002",
    ),
    _paper(
        "test.003",
        "Layered oxide cathodes with suppressed phase transitions",
        "P2-type layered oxides were doped with magnesium to suppress phase "
        "transitions during cycling. Doped cathodes retained 88% capacity "
        "after 300 cycles, compared to 62% for the undoped control.",
        ["D. Author"], "2026-09-02", "OpenAlex", "10.1000/test.003",
    ),
    _paper(
        "test.004",
        "Prussian blue analogues as low-cost sodium cathodes",
        "Prussian blue analogue cathodes were synthesized via a coprecipitation "
        "route. The materials delivered 120 mAh/g at 1C and showed stable "
        "cycling over 1000 cycles with 85% capacity retention.",
        ["E. Author", "F. Author"], "2026-09-03", "OpenAlex", "10.1000/test.004",
    ),
    _paper(
        "test.005",
        "Operando XRD of sodium intercalation in hard carbon",
        "Operando X-ray diffraction was used to track sodium intercalation in "
        "hard carbon anodes. The data support a two-stage storage mechanism "
        "involving adsorption at defects followed by pore filling.",
        ["G. Author"], "2026-09-03", "arXiv", "10.1000/test.005",
    ),
    _paper(
        "test.006",
        "Solid-state sodium electrolytes based on sulfide glasses",
        "Sulfide glass electrolytes were prepared with varying Na2S content. "
        "Ionic conductivity reached 1.2 mS/cm at room temperature, with "
        "stable plating and stripping over 200 cycles.",
        ["H. Author"], "2026-09-04", "OpenAlex", "10.1000/test.006",
    ),
    _paper(
        "test.007",
        "Aqueous sodium-ion batteries with wide voltage windows",
        "A water-in-salt electrolyte enabled a 2.5 V aqueous sodium-ion cell. "
        "The full cell retained 79% capacity after 500 cycles.",
        ["I. Author", "J. Author"], "2026-09-04", "arXiv", "10.1000/test.007",
    ),
    _paper(
        "test.008",
        "Machine-learned interatomic potentials for Na diffusion",
        "A machine-learned interatomic potential was trained on DFT data to "
        "simulate sodium diffusion in layered oxides. Predicted migration "
        "barriers agree with experiment to within 30 meV.",
        ["K. Author"], "2026-09-05", "arXiv", "10.1000/test.008",
    ),
    _paper(
        "test.009",
        "Recycling of sodium-ion battery cathodes",
        "A hydrometallurgical route was developed to recover transition metals "
        "from spent sodium-ion cathodes. Recovery yields exceeded 95% for "
        "nickel, manganese, and iron.",
        ["L. Author"], "2026-09-05", "OpenAlex", "10.1000/test.009",
    ),
    _paper(
        "test.010",
        "Thermal stability of sodium-ion cells at elevated temperatures",
        "Accelerated aging tests at 60 C were performed on sodium-ion pouch "
        "cells. Capacity fade was correlated with electrolyte decomposition "
        "products identified by GC-MS.",
        ["M. Author", "N. Author"], "2026-09-06", "OpenAlex", "10.1000/test.010",
    ),
    _paper(
        "test.011",
        "Tin-based alloy anodes for sodium storage",
        "Tin nanoparticles embedded in a carbon matrix were tested as sodium "
        "anodes. The composite delivered 450 mAh/g over 200 cycles with "
        "limited volume expansion due to the carbon buffer.",
        ["O. Author"], "2026-09-06", "arXiv", "10.1000/test.011",
    ),
    _paper(
        "test.012",
        "Ionic liquid electrolytes for high-voltage sodium cells",
        "A pyrrolidinium-based ionic liquid electrolyte was evaluated in "
        "sodium half-cells. The electrolyte enabled stable cycling to 4.5 V "
        "versus Na/Na+ with limited oxidative decomposition.",
        ["P. Author"], "2026-09-07", "OpenAlex", "10.1000/test.012",
    ),
    _paper(
        "test.013",
        "Sodium-ion battery pack design for grid storage",
        "A 1 kWh sodium-ion battery pack was designed and tested for grid "
        "storage duty cycles. Round-trip efficiency reached 88%.",
        ["Q. Author", "R. Author"], "2026-09-07", "OpenAlex", "10.1000/test.013",
    ),
    _paper(
        "test.014",
        "Cryo-EM of the sodium solid-electrolyte interphase",
        "Cryogenic electron microscopy was used to image the solid-electrolyte "
        "interphase on hard-carbon anodes. The SEI was found to be amorphous "
        "with embedded sodium fluoride nanocrystals.",
        ["S. Author"], "2026-09-08", "arXiv", "10.1000/test.014",
    ),
    _paper(
        "test.015",
        "Techno-economic analysis of sodium-ion battery production",
        "A bottom-up cost model for sodium-ion battery production was "
        "developed. Projected pack-level costs are 15-25% lower than "
        "lithium iron phosphate at equivalent production scale.",
        ["T. Author"], "2026-09-08", "OpenAlex", "10.1000/test.015",
    ),
]


def _banner(msg):
    print("\n" + "=" * 72)
    print(msg)
    print("=" * 72)


def main():
    # ---- Reviewer ------------------------------------------------------
    _banner(f"REVIEWER  ({len(CANDIDATES)} candidates)")
    t0 = time.time()
    selected = None
    try:
        crew = build_review_crew(CANDIDATES, "sodium-ion batteries")
        result = crew.kickoff()
        print(f"Kickoff returned in {time.time() - t0:.1f}s")
        raw = getattr(result, "raw", None)
        if raw is not None:
            print(f"--- raw (first 800 chars) ---\n{raw[:800]}")
        parsed = _parse_output(result, ReviewerSelection)
        print(f"_parse_output -> {'OK' if parsed else 'FAILED (returned None)'}")
        if parsed:
            print(f"  selected_papers count: {len(parsed.selected_papers)}")
            for p in parsed.selected_papers:
                print(f"    id={p.id!r}  reason={p.reason[:90]!r}")
    except Exception:
        traceback.print_exc()
        print("Reviewer stage crashed — stopping here.")
        return

    if not parsed or not parsed.selected_papers:
        print("\nReviewer returned no usable picks — skipping Writer and Fact Checker.")
        return

    # ---- Writer --------------------------------------------------------
    _banner("WRITER")
    from src.crew_setup import select_papers
    selected = select_papers(CANDIDATES, parsed)
    if not selected:
        print("All reviewer picks were hallucinated IDs — skipping Writer.")
        return
    t0 = time.time()
    wparsed = None
    try:
        crew = build_write_crew(selected, "sodium-ion batteries")
        result = crew.kickoff()
        print(f"Kickoff returned in {time.time() - t0:.1f}s")
        raw = getattr(result, "raw", None)
        if raw is not None:
            print(f"--- raw (first 800 chars) ---\n{raw[:800]}")
        wparsed = _parse_output(result, WriterOutput)
        print(f"_parse_output -> {'OK' if wparsed else 'FAILED (returned None)'}")
        if wparsed:
            print(f"  articles count: {len(wparsed.articles)} (expected {len(selected)})")
            for a in wparsed.articles:
                print(f"    id={a.id!r}  claims={len(a.claims)}  title={a.title[:60]!r}")
    except Exception:
        traceback.print_exc()
        print("Writer stage crashed — stopping here.")
        return

    if not wparsed or not wparsed.articles:
        print("\nWriter returned no usable articles — skipping Fact Checker.")
        return

    # ---- Fact checker --------------------------------------------------
    _banner("FACT CHECKER")
    t0 = time.time()
    try:
        report = run_fact_check(wparsed, selected)
        print(f"run_fact_check returned in {time.time() - t0:.1f}s")
        print(f"overall_status: {report.overall_status}")
        print(f"assessments: {len(report.assessments)}")
        for a in report.assessments:
            print(f"  [{a.verdict}] {a.claim[:70]!r}")
    except Exception:
        traceback.print_exc()
        print("Fact checker crashed.")


if __name__ == "__main__":
    main()