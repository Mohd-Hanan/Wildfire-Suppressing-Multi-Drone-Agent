from wildfire.simulation.world import World
from wildfire.simulation.drone import Drone, DroneType

class ActionExecutor:
    """
    Action Mapping:
    0 = Stay
    1 = North (dy = -1)
    2 = South (dy = +1)
    3 = East (dx = +1)
    4 = West (dx = -1)
    5 = Drop Water
    6 = Drop Retardant
    """
    
    @staticmethod
    def execute(drone: Drone, world: World, action_id: int):
        if not drone.active:
            return

        if action_id == 0:
            pass # Stay
        elif action_id == 1:
            drone.move(0, -1, world.width, world.height)
        elif action_id == 2:
            drone.move(0, 1, world.width, world.height)
        elif action_id == 3:
            drone.move(1, 0, world.width, world.height)
        elif action_id == 4:
            drone.move(-1, 0, world.width, world.height)
        elif action_id == 5:
            if drone.type == DroneType.WATER:
                if drone.drop():
                    world.terrain.moisture[drone.x, drone.y] = 1.0
        elif action_id == 6:
            if drone.type == DroneType.RETARDANT:
                if drone.drop():
                    world.terrain.fuel[drone.x, drone.y] = 0.0
