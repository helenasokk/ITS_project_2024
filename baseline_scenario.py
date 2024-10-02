from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging

def main():
    set_up_simple_logging()

    # Specify paths to your BeamNG.tech installation and user folder
    beamng_home = r'C:\\ITS\\BeamNG.tech.v0.32.5.0'
    user_folder = r'C:\\Users\\kaspe\\AppData\\Local\\BeamNG.drive\\0.32'

    # Start up the simulator.
    bng = BeamNGpy('localhost', 64256, home=beamng_home, user=user_folder)
    bng.open(launch=True)

    # Create the scenario in the 'smallgrid' map.
    scenario = Scenario('smallgrid', 'AI_Car_Test')

    # Create the ego vehicle
    ego = Vehicle('ego_vehicle', model='etk800', license='EGO', color='Red')
    scenario.add_vehicle(ego, pos=(0, 0, 0), rot_quat=(0, 0, 0, 1))

    # Create an AI vehicle to act as an obstacle
    ai_vehicle = Vehicle('ai_vehicle', model='etk800', license='AI_CAR', color='Blue')
    scenario.add_vehicle(ai_vehicle, pos=(0, -10, 0), rot_quat=(0, 0, 0, 1))  # Position it 10 meters in front

    # Make the scenario
    scenario.make(bng)

    # Load and start the scenario
    bng.settings.set_deterministic(60)
    bng.scenario.load(scenario)
    bng.scenario.start()

    # Set AI modes
    ego.ai.set_mode('disabled')  # Disable AI for the ego vehicle
    ai_vehicle.ai.set_mode('span')  # Allow the AI vehicle to move around

    print("AI vehicle is active. Press Ctrl+C to exit.")

    # Run the simulation
    try:
        while True:
            bng.control.step(10)  # Step the simulation
    except KeyboardInterrupt:
        print("Simulation stopped by the user.")
    finally:
        bng.close()  # Close BeamNG connection

if __name__ == '__main__':
    main()
