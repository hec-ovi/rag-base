"""Synthetic corpus: 'Halpern Astride Industries' (HAI) fictional world.

Every proper noun, project, theorem, location and dollar figure here is invented
for the sole purpose of being NOT in any model's training data. If a reranker
ranks the right doc top, it's because the model is doing reading comprehension
on the query, not pattern-matching from memorization.

Each doc has a stable id we use to score retrieval (hit@1, hit@5, MRR).
Queries are crafted to be paraphrases or multi-hop, never copy-paste of doc text.
"""

CORPUS = [
    {
        "id": "hai-overview",
        "title": "Halpern Astride Industries: Company Overview",
        "content": (
            "Halpern Astride Industries (HAI) is an aerospace and surface-transit contractor "
            "headquartered in Wallinghurst. HAI runs three flagship programs: Vexil-7 "
            "(autonomous rotor drones), Bramwell (urban transit), and Stilwater (underwater "
            "telemetry). The company was founded in 1962 by Almira Halpern and its current "
            "CTO is Marigold Pemberton. Annual revenue is reported at 184 million orobars."
        ),
    },
    {
        "id": "vexil-7",
        "title": "Project Vexil-7 Program Brief",
        "content": (
            "Vexil-7 is HAI's research drone fleet program. The fleet operates out of "
            "Drumcastle Site B and consists of 28 quadrotor airframes equipped with the "
            "Mirahl-class sensor stack. Each airframe weighs 6.4 kilograms loaded and uses "
            "the Quibbler-Frame protocol for telemetry. Sensor fusion is grounded in "
            "Zigast's theorem, which gives a closed-form bound on multi-hypothesis tracking "
            "error under correlated noise. The program lead is Dr. Penelope Quistain."
        ),
    },
    {
        "id": "bramwell",
        "title": "Project Bramwell: Urban Transit Initiative",
        "content": (
            "Bramwell is HAI's urban transit program. Test track operations run at Pentlow "
            "Yard. Bramwell rolling stock uses the Fenrick coupling for inter-car mechanical "
            "linkage and the Quibbler-Frame protocol for control bus messaging. The program "
            "director is Hardin Volkenburg. Bramwell achieved its first 12-car revenue run "
            "on 2024-11-04, with peak observed velocity of 142 kilometers per hour."
        ),
    },
    {
        "id": "stilwater",
        "title": "Project Stilwater: Underwater Telemetry Program",
        "content": (
            "Stilwater is HAI's underwater telemetry research program. Field operations are "
            "based at Hattersley Reach coastal lab. Stilwater nodes transmit at 38 kilohertz "
            "through saline column using the Hesketh code, an error-correction scheme tuned "
            "for low-bandwidth acoustic links. The chief scientist is Dr. Theron Ashbeck. "
            "Stilwater's first long-baseline trial covered 41 kilometers of coastal shelf "
            "between Hattersley Reach and the Forland buoy."
        ),
    },
    {
        "id": "quistain-bio",
        "title": "Dr. Penelope Quistain: Profile",
        "content": (
            "Dr. Penelope Quistain is the lead engineer of HAI's drone research fleet. She "
            "specializes in quaternion-based attitude estimation and is the author of the "
            "open-form derivation that brings Zigast's theorem within the practical range of "
            "embedded sensor fusion. Quistain joined HAI in 2018 from the Wexbridge "
            "aeronautics lab. She holds a doctorate in nonlinear estimation from Pollard "
            "Institute."
        ),
    },
    {
        "id": "pemberton-bio",
        "title": "Marigold Pemberton: Profile",
        "content": (
            "Marigold Pemberton is the Chief Technology Officer of Halpern Astride "
            "Industries. She joined the executive team in 2020 after twelve years at the "
            "Wallinghurst university research office. Pemberton chairs HAI's standards "
            "council, which steers the Quibbler-Frame protocol revision cycle. Under her "
            "tenure, HAI has consolidated its three flagship programs into a single "
            "engineering pipeline."
        ),
    },
    {
        "id": "volkenburg-bio",
        "title": "Hardin Volkenburg: Profile",
        "content": (
            "Hardin Volkenburg is the program director of HAI's urban transit initiative. "
            "He oversees daily operations at Pentlow Yard. Before joining HAI in 2017, "
            "Volkenburg ran rolling-stock validation at the Glaverstown rail authority. "
            "He is a strong advocate for the Fenrick coupling, which he helped specify "
            "during his Glaverstown years."
        ),
    },
    {
        "id": "ashbeck-bio",
        "title": "Dr. Theron Ashbeck: Profile",
        "content": (
            "Dr. Theron Ashbeck leads HAI's underwater telemetry program from the "
            "Hattersley Reach coastal lab. His research interests include shallow-water "
            "acoustic propagation and low-power error correction, including the Hesketh "
            "code. Ashbeck previously held a tenured position at Pollard Institute, where "
            "he supervised Penelope Quistain's doctoral work on related estimation problems."
        ),
    },
    {
        "id": "quibbler-frame",
        "title": "The Quibbler-Frame Protocol Specification",
        "content": (
            "The Quibbler-Frame protocol is HAI's internal data-exchange standard. Frames "
            "are 256 bytes wide, prefixed by a 6-byte sentinel and a 2-byte sequence "
            "counter. The protocol assumes a half-duplex bus and tolerates up to 200 "
            "milliseconds of round-trip jitter. Quibbler-Frame is used by both the Vexil-7 "
            "drone telemetry stack and the Bramwell control bus."
        ),
    },
    {
        "id": "zigast-theorem",
        "title": "Zigast's Theorem: Application Note",
        "content": (
            "Zigast's theorem states that the steady-state covariance of a multi-hypothesis "
            "tracker under correlated process noise is bounded by an explicit function of "
            "the noise correlation length. The theorem first appeared in Zigast's 1991 "
            "monograph and was unused in industry until Dr. Penelope Quistain's 2021 "
            "open-form derivation made it computationally practical. Today it underpins the "
            "Vexil-7 sensor fusion pipeline."
        ),
    },
    {
        "id": "fenrick-coupling",
        "title": "The Fenrick Coupling: Technical Reference",
        "content": (
            "The Fenrick coupling is a passive mechanical interface between rolling stock "
            "cars. It provides a self-aligning conical mating surface and a 14-millimeter "
            "shear-pin failure mode. Originally specified by Hardin Volkenburg at the "
            "Glaverstown rail authority, the Fenrick coupling is the standard mechanical "
            "linkage in HAI's Bramwell rolling stock."
        ),
    },
    {
        "id": "hesketh-code",
        "title": "The Hesketh Code: Error-Correction Scheme",
        "content": (
            "The Hesketh code is a low-rate convolutional error-correction scheme designed "
            "for narrow-band acoustic links. It uses a constraint length of 9 and a "
            "puncturing pattern that drops every fourth parity bit, giving an effective "
            "rate of 3/8. Hesketh-coded packets are robust against the long fading bursts "
            "characteristic of shallow-water acoustic channels. The Stilwater program at "
            "Hattersley Reach uses it as its standard link layer."
        ),
    },
    {
        "id": "wallinghurst-hq",
        "title": "Wallinghurst Headquarters Memorandum 2024-Q4",
        "content": (
            "The Wallinghurst headquarters of Halpern Astride Industries hosts the "
            "executive team, the Quibbler-Frame standards council, and the inter-program "
            "engineering review board. Wallinghurst is a port city on the eastern coast. "
            "The headquarters building was completed in 2009 and houses approximately 410 "
            "engineering staff."
        ),
    },
    {
        "id": "drumcastle-ops",
        "title": "Drumcastle Site B: Operations Status",
        "content": (
            "Drumcastle Site B is the airfield from which HAI's Vexil-7 drone fleet "
            "operates. It lies 64 kilometers inland from Wallinghurst and shares its main "
            "runway with the Drumcastle municipal airport under a tenancy agreement signed "
            "in 2019. As of the last operations review, 24 of 28 airframes were flight-ready."
        ),
    },
    {
        "id": "pentlow-yard",
        "title": "Pentlow Yard: Test Track Plan",
        "content": (
            "Pentlow Yard is the dedicated test track for HAI's Bramwell rolling stock. "
            "It comprises a 4.2-kilometer mainline loop, a 600-meter siding, and a "
            "platform mock-up for boarding trials. Daily operations are supervised by "
            "Hardin Volkenburg's program office. The yard adjoins the Pentlow river "
            "embankment to the west."
        ),
    },
    {
        "id": "hattersley-reach",
        "title": "Hattersley Reach Coastal Lab",
        "content": (
            "Hattersley Reach is HAI's coastal research facility, where the Stilwater "
            "underwater telemetry program runs its field trials. The lab maintains a "
            "fleet of three submersible relay nodes and a shore-side acoustic monitoring "
            "station. Field campaigns are coordinated with Forland buoy survey runs by "
            "Dr. Theron Ashbeck."
        ),
    },
    {
        "id": "distractor-1",
        "title": "Wexbridge Aeronautics Lab: Annual Bulletin",
        "content": (
            "The Wexbridge aeronautics lab is an academic research center unrelated to "
            "Halpern Astride Industries except through staff overlap. Wexbridge runs "
            "wind-tunnel experiments and publishes a quarterly bulletin on subsonic "
            "boundary layer phenomena. Their tools include the Crackmore vortex generator "
            "and the Pendrith hot-wire calibration kit."
        ),
    },
    {
        "id": "distractor-2",
        "title": "Glaverstown Rail Authority: Public Schedule",
        "content": (
            "The Glaverstown rail authority is a regional public transport operator. Its "
            "fleet consists of legacy diesel multiple units and three newer light-rail "
            "tram lines. Passenger volume in 2024 was 8.3 million annual rides. The "
            "authority is unaffiliated with Halpern Astride Industries, although former "
            "Glaverstown engineer Hardin Volkenburg now runs HAI's Bramwell program."
        ),
    },
    {
        "id": "distractor-3",
        "title": "Pollard Institute: Doctoral Program Brochure",
        "content": (
            "The Pollard Institute offers doctoral programs in applied mathematics, "
            "estimation theory, and shallow-water acoustics. Its alumni include both "
            "Dr. Penelope Quistain and Dr. Theron Ashbeck, currently at HAI. The institute "
            "itself does no aerospace work and has no programmatic relationship with "
            "Halpern Astride Industries."
        ),
    },
    {
        "id": "distractor-4",
        "title": "Forland Buoy Survey: Coastal Hydrography Note",
        "content": (
            "The Forland buoy is a fixed oceanographic station 41 kilometers offshore from "
            "Hattersley Reach. It logs water temperature, salinity, and current direction. "
            "The buoy is operated by the regional hydrographic service, not by HAI, "
            "though HAI's Stilwater program coordinates joint survey runs with the buoy "
            "operators."
        ),
    },
]

# Each query targets ONE specific doc. The query is a paraphrase or a multi-hop
# question, never a copy-paste of the doc title or text.
QUERIES = [
    # Paraphrase
    {"q": "the drone fleet research program at HAI", "target": "vexil-7"},
    {"q": "underwater data telemetry initiative", "target": "stilwater"},
    {"q": "HAI's urban transit program", "target": "bramwell"},
    # Exact-phrase rare jargon
    {"q": "Quibbler-Frame protocol details", "target": "quibbler-frame"},
    {"q": "Hesketh code error correction", "target": "hesketh-code"},
    {"q": "Zigast's theorem applications", "target": "zigast-theorem"},
    # Entity lookup
    {"q": "who is the chief technology officer of Halpern Astride Industries",
     "target": "pemberton-bio"},
    {"q": "lead engineer for the Vexil-7 sensor fusion work", "target": "quistain-bio"},
    # Multi-hop
    {"q": "where does the Bramwell program director run his daily operations",
     "target": "pentlow-yard"},  # Volkenburg -> Bramwell -> Pentlow
    {"q": "the coastal lab where Dr. Ashbeck's program operates",
     "target": "hattersley-reach"},
    # Polysemy / distractor pressure
    {"q": "company headquarters on the eastern coast",
     "target": "wallinghurst-hq"},
    {"q": "airfield used by HAI's drone fleet", "target": "drumcastle-ops"},
]
