from pysc2.agents import base_agent
from pysc2.lib import actions
import numpy as np

"""
======================================================
self.step : 24
obs type : <class 'pysc2.env.environment.TimeStep'>
------------------------------------------------------
obs.step_type : <enum 'StepType'>
StepType.MID    [FIRST, MID, LAST]
------------------------------------------------------
obs.reward : <class 'int'>
0
------------------------------------------------------
obs.discount : <class 'float'>
1.0
------------------------------------------------------
obs.observation : <class 'dict'> - {str : numpy.ndarray}
obs.observation['build_queue'] : (n, 7)
    ..['build_queue'][i][j] : same as single_select
obs.observation['game_loop'] : (1,)
obs.observation['cargo_slots_available'] : (1,)
obs.observation['player'] : (11,)
    ..['player'][0] : player_id
    ..['player'][1] : mineral
    ..['player'][2] : vespine
    ..['player'][3] : food used
    ..['player'][4] : food cap
    ..['player'][5] : food used by army
    ..['player'][6] : food used by workers
    ..['player'][7] : idle worker count
    ..['player'][8] : army count
    ..['player'][9] : warp gate count
    ..['player'][10] : larva count
obs.observation['available_actions'] : (n)
    ..['available_actions'][i] : available action id
obs.observation['minimap'] : (7, 64, 64)
    ..['minimap'][0] : height_map
    ..['minimap'][1] : visibility
    ..['minimap'][2] : creep
    ..['minimap'][3] : camera
    ..['minimap'][4] : player_id
    ..['minimap'][5] : player_relative              < [0,4] < [background, self, ally, neutral, enemy]
    ..['minimap'][6] : selected                     < 0 for not selected, 1 for selected
obs.observation['cargo'] : (n, 7) - n is the number of all units in a transport
    ..['cargo'][i][j] :  same as single_select[0][j]
obs.observation['multi_select'] : (n, 7)
    ..['multi_select'][i][j] : same as single_select[0][j]
        -> single_select 과 양존하지 않음.
           single_select시엔 multi_select=[]
           multi_select 시엔 single_select = [[0,0,0,0,0,0,0]]
obs.observation['score_cumulative'] : (13,)
obs.observation['control_groups'] : (10, 2)
    ..['control_groups'][i][0] : i'th unit leader type
    ..['control_groups'][i][1] : count
obs.observation['single_select'] : (1, 7)
    ..['single_select'][0][0] : unit_type
    ..['single_select'][0][1] : player_relative     < [0,4] < [background, self, ally, neutral, enemy]
    ..['single_select'][0][2] : health
    ..['single_select'][0][3] : shields
    ..['single_select'][0][4] : energy
    ..['single_select'][0][5] : transport slot
    ..['single_select'][0][6] : build progress as percentage
obs.observation['screen'] : (13, 84, 84)
    ..['screen'][0] : height_map
    ..['screen'][1] : visibility
    ..['screen'][2] : creep
    ..['screen'][3] : power                         < protoss power
    ..['screen'][4] : player_id
    ..['screen'][5] : player_relative               < [0,4] < [background, self, ally, neutral, enemy]
    ..['screen'][6] : unit_type
    ..['screen'][7] : selected                      < 0 for not selected, 1 for selected
    ..['screen'][8] : hit_points
    ..['screen'][9] : energy
    ..['screen'][10] : shields
    ..['screen'][11] : unit_density
    ..['screen'][12] : unit_density_aa
======================================================
"""

STATE_SIZE = 23
# observation to state number
def State(obs):
    ans = np.zeros([1, STATE_SIZE])
    ans[0][0] = obs['player'][1]
    ans[0][1] = obs['player'][2]
    ans[0][2] = obs['player'][3]
    ans[0][3] = obs['player'][4]
    ans[0][4] = obs['player'][5]
    ans[0][5] = obs['player'][6]
    ans[0][6] = obs['player'][7]
    ans[0][7] = obs['player'][8]
    ans[0][8] = obs['player'][8]
    ans[0][9] = obs['player'][10]
    return ans

class MinervaAgent(base_agent.BaseAgent):
    def setup(self, obs_spec, action_spec):
        super(MinervaAgent, self).setup(obs_spec, action_spec)
        self.mysingleselect = []

    def step(self, obs, Qs):
        super(MinervaAgent, self).step(obs)
        if not np.array_equal(obs.observation['player'], self.mysingleselect):
            self.mysingleselect = obs.observation['player']

        # if Qs == 0, choose an action for exploration
        if type(Qs) == int and Qs == 0:
            function_id = np.random.choice(obs.observation["available_actions"])
            args = [[np.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
            return actions.FunctionCall(function_id, args)

        # TODO : 
        # otherwise choose an action for exploit
        return actions.FunctionCall(0, [])
