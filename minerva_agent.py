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

def intToCoordinate(num, size=64):
    if size!=64:
        num = num * size * size // 4096
    y = num // size
    x = num - size * y
    return [x, y]

class MinervaAgent(base_agent.BaseAgent):
    def __init__(self, mainDQN=None):
        super(MinervaAgent, self).__init__()
        self.mainDQN = mainDQN

    def setup(self, obs_spec, action_spec):
        super(MinervaAgent, self).setup(obs_spec, action_spec)

    def step(self, obs, exploit):
        super(MinervaAgent, self).step(obs)

        # if exploit == 0, choose an action for exploration
        if exploit == 0:
            ans_id = np.random.choice(obs.observation["available_actions"])

        ################## find action id ######################

        # otherwise choose an action for exploit
        # Qs[0] : ndarray([584]) -> Qs[0][i] score function of action whose id=i
        else:
            Qs = self.mainDQN.predict([[obs.observation]])
            for i in range(len(Qs[0])):
                if i not in obs.observation["available_actions"]:
                    Qs[0][i] = -100

            ans_id = np.argmax(Qs[0])
            if Qs[0][ans_id] <= -100:
                ans_id = 0

        ############# find minimap/screen coordinate etc. #################

        spatialQs = self.mainDQN.predictSpatial([[obs.observation]])
        spatialInt = []
        for i in range(13):
            spatialInt.append(np.argmax(spatialQs[i][0]))

        ans_arg = []
        for arg in self.action_spec.functions[ans_id].args:
            if arg.id in range(3):
                ans_arg.append(intToCoordinate(spatialInt[arg.id], arg.sizes[0]))
            else:
                ans_arg.append([spatialInt[arg.id]])

        print("step :", self.steps, "exploit :",exploit,"action id :", ans_id, "arg :",ans_arg)
        """
173   Attributes:
174     0  screen: A point on the screen.
175     1  minimap: A point on the minimap.
176     2  screen2: The second point for a rectangle. This is needed so that no
177          function takes the same type twice.
178     3  queued: Whether the action should be done now or later.                 size<2
179     4  control_group_act: What to do with the control group.                   size<5
180     5  control_group_id: Which control group to do it with.                    size<10
181     6  select_point_act: What to do with the unit at the point.                size<4
182     7  select_add: Whether to add the unit to the selection or replace it.     size<2
183     8  select_unit_act: What to do when selecting a unit by id.                size<4
184     9  select_unit_id: Which unit to select by id.                             size<500
185     10 select_worker: What to do when selecting a worker.                      size<4
186     11 build_queue_id: Which build queue index to target.                      size<10
187     12 unload_id: Which unit to target in a transport/nydus/command center.    size<500
        """
        return actions.FunctionCall(ans_id, ans_arg)
