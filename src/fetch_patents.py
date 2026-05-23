import json
import os

def load_patents(filename="data/patents.json"):
    """Loads patents from file, creates sample data if file doesn't exist"""
    if not os.path.exists(filename):
        patents = get_sample_patents()
        save_patents(patents)
        return patents
    with open(filename, "r") as f:
        patents = json.load(f)
    print(f"Loaded {len(patents)} patents from file")
    return patents


def save_patents(patents, filename="data/patents.json"):
    """Saves patents to a JSON file"""
    os.makedirs("data", exist_ok=True)
    with open(filename, "w") as f:
        json.dump(patents, f, indent=2)
    print(f"Saved {len(patents)} patents to {filename}")


def get_sample_patents():
    """15 real aerospace patents — pre-loaded for offline development"""
    return [
        {
            "number": "11584539",
            "title": "Drone delivery system with autonomous navigation",
            "abstract": "A drone delivery system comprising autonomous navigation capabilities for urban environments. The system includes obstacle detection, path planning algorithms, and fail-safe landing procedures for unmanned aerial vehicles operating in controlled airspace.",
            "date": "2023-02-21",
            "year": "2023"
        },
        {
            "number": "11440671",
            "title": "Aerospace propulsion system with variable thrust control",
            "abstract": "An aerospace propulsion system incorporating variable thrust vector control for improved maneuverability. The system uses advanced combustion chamber design and nozzle geometry optimization to maximize fuel efficiency in high-altitude flight conditions.",
            "date": "2022-09-13",
            "year": "2022"
        },
        {
            "number": "11319068",
            "title": "UAV swarm coordination using distributed AI algorithms",
            "abstract": "A method and system for coordinating swarms of unmanned aerial vehicles using distributed artificial intelligence. Each UAV runs local decision-making algorithms while communicating with neighboring units to achieve collective mission objectives without centralized control.",
            "date": "2022-05-03",
            "year": "2022"
        },
        {
            "number": "11267579",
            "title": "Satellite attitude control system using reaction wheels",
            "abstract": "An attitude control system for low-earth orbit satellites employing reaction wheels and magnetic torquers. The system incorporates predictive control algorithms to maintain precise orbital positioning and minimize fuel consumption during station-keeping maneuvers.",
            "date": "2022-03-08",
            "year": "2022"
        },
        {
            "number": "11186358",
            "title": "Electric vertical takeoff and landing aircraft propulsion",
            "abstract": "An electric propulsion system for vertical takeoff and landing aircraft featuring distributed motor architecture. The system provides redundancy through multiple independent propulsion units and uses machine learning to optimize power distribution across all motors during flight.",
            "date": "2021-11-30",
            "year": "2021"
        },
        {
            "number": "11142330",
            "title": "Hypersonic vehicle thermal protection system",
            "abstract": "A thermal protection system for hypersonic aerospace vehicles designed to withstand temperatures exceeding 2000 degrees Celsius. The system uses advanced ceramic composite materials with embedded cooling channels to manage heat flux during atmospheric reentry.",
            "date": "2021-10-12",
            "year": "2021"
        },
        {
            "number": "11091257",
            "title": "Autonomous drone inspection system for aircraft maintenance",
            "abstract": "An autonomous drone-based inspection system for commercial aircraft maintenance. The system uses computer vision and deep learning to detect surface defects, corrosion, and structural anomalies, generating automated maintenance reports with precise defect location mapping.",
            "date": "2021-08-17",
            "year": "2021"
        },
        {
            "number": "11027854",
            "title": "Aerospace fuel cell power generation system",
            "abstract": "A hydrogen fuel cell power generation system for aerospace applications providing clean auxiliary power. The system integrates with existing aircraft electrical architecture and uses advanced membrane electrode assemblies optimized for high-altitude low-pressure operation.",
            "date": "2021-06-08",
            "year": "2021"
        },
        {
            "number": "10974827",
            "title": "Machine learning based flight path optimization for UAV",
            "abstract": "A machine learning system for real-time flight path optimization of unmanned aerial vehicles. The system uses reinforcement learning to adapt to changing weather conditions, air traffic, and mission requirements while minimizing energy consumption and flight time.",
            "date": "2021-04-13",
            "year": "2021"
        },
        {
            "number": "10926867",
            "title": "Aerospace composite wing structure with embedded sensors",
            "abstract": "A composite wing structure for aerospace vehicles with embedded fiber optic sensors for structural health monitoring. The system continuously measures strain, temperature, and vibration data to predict fatigue failure and schedule preventive maintenance interventions.",
            "date": "2021-02-23",
            "year": "2021"
        },
        {
            "number": "10870494",
            "title": "Drone traffic management system for urban air mobility",
            "abstract": "A comprehensive traffic management system for drone operations in urban environments. The system uses AI-based conflict detection and resolution algorithms to safely coordinate thousands of simultaneous drone flights in low-altitude urban airspace.",
            "date": "2020-12-22",
            "year": "2020"
        },
        {
            "number": "10814978",
            "title": "Spacecraft propulsion using ion thruster arrays",
            "abstract": "An advanced spacecraft propulsion system using arrays of ion thrusters for deep space missions. The system provides precise thrust vectoring and extremely high specific impulse by accelerating xenon ions through an electromagnetic grid structure.",
            "date": "2020-10-27",
            "year": "2020"
        },
        {
            "number": "10759543",
            "title": "AI powered predictive maintenance for jet engines",
            "abstract": "An artificial intelligence system for predictive maintenance of commercial jet engines. The system analyzes real-time sensor data from thousands of engine parameters to predict component failures weeks in advance, significantly reducing unplanned maintenance events.",
            "date": "2020-09-01",
            "year": "2020"
        },
        {
            "number": "10703508",
            "title": "Solar powered high altitude pseudo satellite platform",
            "abstract": "A solar powered unmanned aerial vehicle designed for high altitude long endurance operation as a pseudo satellite. The platform uses lightweight photovoltaic cells and energy storage systems to maintain continuous flight at stratospheric altitudes for months.",
            "date": "2020-07-07",
            "year": "2020"
        },
        {
            "number": "10647438",
            "title": "Adaptive wing morphing system for aerospace efficiency",
            "abstract": "An adaptive wing morphing system that dynamically changes wing geometry during flight to optimize aerodynamic efficiency. The system uses shape memory alloy actuators controlled by real-time optimization algorithms responding to flight condition changes.",
            "date": "2020-05-12",
            "year": "2020"
        }
    ]


if __name__ == "__main__":
    patents = get_sample_patents()
    save_patents(patents)

    print("\n--- Patent Data Ready ---")
    for i, p in enumerate(patents[:3]):
        print(f"\n[{i+1}] {p['title']}")
        print(f"    Number: {p['number']} | Date: {p['date']}")
        print(f"    Abstract: {p['abstract'][:120]}...")

    print(f"\nTotal: {len(patents)} patents ready for AI processing")