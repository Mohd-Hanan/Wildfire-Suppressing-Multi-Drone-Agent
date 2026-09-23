import yaml
from wildfire.simulation.drone import Drone, DroneType
from wildfire.simulation.world import World
from wildfire.environment.action import ActionExecutor

def main():
    print("========================================")
    print("STAGE 7.21 — REFILL BEHAVIOR TEST")
    print("========================================")
    
    world = World(48, 48, seed=999)
    drone = world.drones[0]
    
    print("1. Initial state:")
    print(f"   Drone 0 type: {drone.type.name}")
    print(f"   Position: ({drone.x},{drone.y}) | Base: ({world.base_x},{world.base_y})")
    print(f"   Battery: {drone.battery} / {drone.max_battery}")
    print(f"   Payload: {drone.payload} / {drone.max_payload}")
    
    print("\n2. Consuming resources (moving away, dropping 5 water)...")
    ActionExecutor.execute(drone, world, 1) # North (away from base)
    for _ in range(5):
        ActionExecutor.execute(drone, world, 5) # Drop Water
        
    print(f"   Position: ({drone.x},{drone.y})")
    print(f"   Battery: {drone.battery} (expected 150 - 1 - 5*2 = 139)")
    print(f"   Payload: {drone.payload} (expected 5 - 5*1 = 0)")
    
    print("\n3. Returning to base...")
    ActionExecutor.execute(drone, world, 2) # South (back to base)
    print(f"   Position: ({drone.x},{drone.y})")
    print(f"   is_at_base: {world.is_at_base(drone)}")
    
    print("\n4. Processing base refills...")
    world.process_base_refills()
    print(f"   Battery: {drone.battery} / {drone.max_battery}")
    print(f"   Payload: {drone.payload} / {drone.max_payload}")
    
    if drone.battery == 150 and drone.payload == 5:
        print("\nTEST PASSED: Refill mechanics work perfectly.")
    else:
        print("\nTEST FAILED: Refill mechanics are broken.")

if __name__ == "__main__":
    main()
