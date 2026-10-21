"""Demo data. Everything here is SYNTHETIC.

The tender notices are written in the style of real Turkish public procurement notices (EKAP,
the national e-procurement platform), but the institutions, amounts and the bidder company are
invented for the demo.
"""

from __future__ import annotations

import random
from typing import Any

COMPANY_PROFILE: dict[str, Any] = {
    "name": "Atlas Yapı ve Elektrik Ltd. (fictional)",
    "fields": ["building construction", "electrical installation", "renovation of public buildings"],
    "documented_similar_work_experience_try": 18_500_000,
    "certificates": ["ISO 9001", "ISO 14001", "TS EN ISO 45001", "Contractor group B certificate"],
    "missing_certificates": ["Contractor group A certificate", "TSE service adequacy certificate for elevators"],
    "max_bid_bond_cash_try": 1_200_000,
    "min_days_to_prepare_bid": 7,
    "operating_regions": ["Istanbul", "Kocaeli", "Bursa", "Ankara"],
}

# Short tender notices for the live demo. `notice_tr` is what the agent reads (Turkish, like the
# real source); the LLM writes an English brief of the requirements before the decision is made.
DEMO_TENDERS: list[dict[str, Any]] = [
    {
        "id": "2026/1184301",
        "title": "School gym roof renovation, Kocaeli",
        "notice_tr": (
            "İhale kayıt no: 2026/1184301. İdare: Kocaeli İl Milli Eğitim Müdürlüğü. "
            "İşin adı: Gebze Anadolu Lisesi spor salonu çatı yenileme işi. Yaklaşık maliyet: 9.800.000 TL. "
            "İhale usulü: Açık ihale. Teklif son tarihi: ihale tarihinden 21 gün sonra. "
            "Yeterlik: İsteklinin teklif bedelinin %50'sinden az olmamak üzere benzer iş deneyimi belgesi sunması "
            "gerekir. Benzer iş: B/III grubu bina işleri. Geçici teminat: teklif bedelinin %3'ü. "
            "ISO 9001 kalite belgesi istenmektedir."
        ),
    },
    {
        "id": "2026/1190027",
        "title": "Hospital elevator modernisation, Ankara",
        "notice_tr": (
            "İhale kayıt no: 2026/1190027. İdare: Ankara İl Sağlık Müdürlüğü. "
            "İşin adı: Etlik Devlet Hastanesi 6 adet asansör modernizasyonu. Yaklaşık maliyet: 14.200.000 TL. "
            "Teklif son tarihi: 18 gün sonra. Yeterlik: Benzer iş deneyimi teklif bedelinin %40'ı. "
            "İstekliler asansör için TSE Hizmet Yeterlilik Belgesi sunmak zorundadır; belge sunmayan "
            "isteklilerin teklifleri değerlendirme dışı bırakılır. Geçici teminat %3."
        ),
    },
    {
        "id": "2026/1201455",
        "title": "Municipal service building construction, Istanbul",
        "notice_tr": (
            "İhale kayıt no: 2026/1201455. İdare: Beylikdüzü Belediyesi. İşin adı: Hizmet binası yapım işi. "
            "Yaklaşık maliyet: 78.000.000 TL. Teklif son tarihi: 25 gün sonra. Yeterlik: Teklif bedelinin "
            "%70'inden az olmamak üzere benzer iş deneyimi. A grubu yüklenicilik yetki belgesi zorunludur. "
            "Geçici teminat %3 (yaklaşık 2.340.000 TL)."
        ),
    },
    {
        "id": "2026/1207718",
        "title": "Courthouse electrical installation renewal, Bursa",
        "notice_tr": (
            "İhale kayıt no: 2026/1207718. İdare: Bursa Adalet Komisyonu Başkanlığı. "
            "İşin adı: Adliye binası elektrik tesisatı yenileme işi. Yaklaşık maliyet: 21.000.000 TL. "
            "Teklif son tarihi: 5 gün sonra. Yeterlik: Teklif bedelinin %80'i kadar benzer iş deneyimi "
            "(elektrik tesisatı işleri). ISO 9001 ve TS EN ISO 45001 belgeleri istenmektedir. Geçici teminat %3."
        ),
    },
]

# --------------------------------------------------------------------------- benchmark set
#
# Labelled briefs for the latency/accuracy comparison. Labels come from simple, explicit rules so
# anyone can check them: a missing mandatory certificate or experience below the requirement is
# no_go; a deadline shorter than the company's minimum preparation time, or experience within 10%
# of the requirement, is human_review; everything else is go.

CERTS_HAVE = ["ISO 9001", "ISO 14001", "TS EN ISO 45001", "Contractor group B certificate"]
CERTS_MISSING = ["Contractor group A certificate", "TSE service adequacy certificate for elevators"]
WORKS = [
    ("school building renovation", "Kocaeli"), ("health centre construction", "Bursa"),
    ("municipal office electrical installation", "Istanbul"), ("library roof repair", "Ankara"),
    ("sports hall construction", "Kocaeli"), ("police station renovation", "Istanbul"),
]


def _brief(work: str, region: str, cost: int, exp_ratio: float, certs: list[str], days: int) -> str:
    required = int(cost * exp_ratio)
    return (
        f"Work: {work} in {region}. Estimated cost: {cost:,} TRY. "
        f"Required similar-work experience: at least {required:,} TRY. "
        f"Mandatory certificates: {', '.join(certs)}. "
        f"Bid bond: 3% of the bid. Days left until the bid deadline: {days}."
    )


def benchmark_cases(n: int = 30, seed: int = 7) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    have_exp = COMPANY_PROFILE["documented_similar_work_experience_try"]
    min_days = COMPANY_PROFILE["min_days_to_prepare_bid"]
    cases: list[dict[str, Any]] = []
    kinds = ["go", "no_go_cert", "review_deadline", "go", "no_go_exp", "review_exp"]
    for i in range(n):
        kind = kinds[i % len(kinds)]
        work, region = rng.choice(WORKS)
        certs = rng.sample(CERTS_HAVE, k=2)
        days = rng.randint(min_days + 5, 30)
        ratio = 0.5
        if kind == "go":
            cost = rng.randrange(8_000_000, 28_000_000, 100_000)  # required <= 14M < 18.5M
        elif kind == "no_go_cert":
            cost = rng.randrange(8_000_000, 28_000_000, 100_000)
            certs = [rng.choice(CERTS_MISSING)] + certs[:1]
        elif kind == "no_go_exp":
            cost = rng.randrange(45_000_000, 90_000_000, 100_000)  # required >= 22.5M > 18.5M
        elif kind == "review_deadline":
            cost = rng.randrange(8_000_000, 28_000_000, 100_000)
            days = rng.randint(2, min_days - 2)
        else:  # review_exp: requirement within 10% below what the company has
            cost = int(have_exp * rng.uniform(0.91, 0.99) / ratio)
        label = {"go": "go", "no_go_cert": "no_go", "no_go_exp": "no_go"}.get(kind, "human_review")
        cases.append({"id": f"B{i + 1:02d}", "kind": kind, "label": label,
                      "brief": _brief(work, region, cost, ratio, certs, days)})
    return cases
