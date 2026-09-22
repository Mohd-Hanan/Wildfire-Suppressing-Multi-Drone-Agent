from gymnasium.utils.env_checker import check_env
from wildfire.environment.wildfire_env import WildfireEnv

print('--- GYMNASIUM CHECKER ---')
env = WildfireEnv()
check_env(env)
print('GYMNASIUM CHECK PASSED')

print('\n--- EXPLICIT DTYPE AND CONTAINMENT CHECK ---')
obs, info = env.reset()
print("spatial dtype =", obs["spatial"].dtype)
print("drone dtype =", obs["drone"].dtype)
print("wind dtype =", obs["wind"].dtype)
print("observation_space.contains(obs) =", env.observation_space.contains(obs))

print('\n--- 7 ACTIONS CONTAINMENT CHECK ---')
all_pass = True
for a in range(7):
    o, r, term, trunc, i = env.step(a)
    c = env.observation_space.contains(o)
    print(f"Action {a} containment: {c}")
    if not c: all_pass = False
print(f"All 7 actions contained = {all_pass}")
