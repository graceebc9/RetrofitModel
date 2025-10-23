import logging 

logger = logging.getLogger(__name__)

def get_intervention_list(wall_type, joint_intervention):
    # Handle both string and list inputs
    if isinstance(joint_intervention, list):
        if len(joint_intervention) == 1:
            joint_intervention = joint_intervention[0]
        elif len(joint_intervention) > 1:
            logger.debug(f'Longer list: {joint_intervention}')
            raise Exception('Why is there a list going in here')
        else:
            raise Exception('Empty list provided')
    # If it's already a string, just use it as-is
    elif not isinstance(joint_intervention, str):
        raise TypeError(f'Expected string or list, got {type(joint_intervention)}')
    
    logger.debug(f"Get intervention : {wall_type},  {joint_intervention} ")
    
    if 'cavity' in wall_type:
        wt = 'cavity_wall_percentile' 
    elif 'internal' in wall_type:
        wt = 'solid_wall_internal_percentile'
    elif 'external' in wall_type:
        wt = 'solid_wall_external_percentile'
    else:
        raise Exception('wall-type not as expected: ', wall_type)
    
    joint_intervention_dict = {
        'joint_loft_wall_add': [wt, 'loft_percentile'], 
        'joint_loft_wall_decay': [wt, 'loft_percentile'], 
        'joint_heat_ins_add': [wt, 'loft_percentile', 'heat_pump_percentile'], 
        'joint_heat_ins_decay': [wt, 'loft_percentile', 'heat_pump_percentile'], 
    } 
                                     
    # If it's a joint intervention, return the list from the dict
    if joint_intervention in joint_intervention_dict:
        interventions_list = joint_intervention_dict[joint_intervention]
    else:
        # If it's a single intervention, just return it as a single-item list
        interventions_list = [joint_intervention]
        logger.debug(f'Single intervention, returning as list: {interventions_list}')
    
    return interventions_list

# def get_intervention_list(wall_type, joint_intervention):
#     # Handle both string and list inputs
#     if isinstance(joint_intervention, list):
#         if len(joint_intervention) == 1:
#             joint_intervention = joint_intervention[0]
#         elif len(joint_intervention) > 1:
#             logger.debug(f'Longer list: {joint_intervention}')
#             raise Exception('Why is there a list going in here')
#         else:
#             raise Exception('Empty list provided')
#     # If it's already a string, just use it as-is
#     elif not isinstance(joint_intervention, str):
#         raise TypeError(f'Expected string or list, got {type(joint_intervention)}')
    
#     logger.debug(f"Get intervention : {wall_type},  {joint_intervention} ")
    
#     if 'cavity' in wall_type:
#         wt = 'cavity_wall_percentile' 
#     elif 'internal' in wall_type:
#         wt = 'solid_wall_internal_percentile'
#     elif 'external' in wall_type:
#         wt = 'solid_wall_external_percentile'
#     else:
#         raise Exception('wall-type not as expected: ', wall_type)
    
#     joint_intervention_dict = {
#         'joint_loft_wall_add': [wt, 'loft_percentile'], 
#         'joint_loft_wall_decay': [wt, 'loft_percentile'], 
#         'joint_heat_ins_add': [wt, 'loft_percentile', 'heat_pump_percentile'], 
#         'joint_heat_ins_decay': [wt, 'loft_percentile', 'heat_pump_percentile'], 
#     } 
                                     
#     if joint_intervention not in joint_intervention_dict.keys():
#         raise KeyError(f'No package found for: {joint_intervention}')
    
#     interventions_list = joint_intervention_dict[joint_intervention]
    
#     return interventions_list


retrofit_packages = {
    
        'loft_installation': {
        'name': 'loft',
        'description': 'using percentiles to get savings',
        'interventions': [
            'loft_percentile',
        ],
        'includes_wall_insulation': False,
        'installation_approach': 'simultaneous'
    },
    'wall_installation': {
        'name': 'wall',
        'description': 'using percentiles to get savings',
        'interventions': [
            'WALL_INSULATION',
        ],
        'includes_wall_insulation': True,
        'installation_approach': 'simultaneous'

    },

    'joint_loft_wall_add': {
        'name': 'wall_and_loft',
        'description': 'combine using multiplicative ',
        'interventions': [
            'joint_loft_wall_add',

        ],
        'includes_wall_insulation': True,
        'installation_approach': 'simultaneous'
    },
    'joint_loft_wall_decay': {
        'name': 'wall_and_loft',
        'description': 'combine using multiplicative ',
        'interventions': [
            'joint_loft_wall_decay',

        ],
        'includes_wall_insulation': True,
        'installation_approach': 'simultaneous'
    },

    'heat_pump_only': 
    {
        'name': 'heat pump only',
        'description': '  ',
        'interventions': [
            'heat_pump_percentile',

        ],
        'includes_wall_insulation': False,
        'installation_approach': 'simultaneous'
    },

    'join_heat_ins_add': 
        {
        'name': 'heat pump and insulation',
        'description': '  ',
        'interventions': [
            'joint_heat_ins_add',

        ],
        'includes_wall_insulation': True,
        'installation_approach': 'simultaneous'
    },
    'join_heat_ins_decay': 
        {
        'name': 'heat pump and all insulation',
        'description': '  ',
        'interventions': [
            'joint_heat_ins_decay',

        ],
        'includes_wall_insulation': True,
        'installation_approach': 'simultaneous'
    },

    'scenario2': {
        'name': 'Scenario 2: envelope only (evans et al)',
        'description': 'Focus on insulation with easy wins',
        'interventions': [
            'WALL_INSULATION',
            'loft_insulation',
            'double_glazing',
        ],
        'includes_wall_insulation': True,
        'installation_approach': 'simultaneous'
    },

    'scenario3': {
        'name': 'envelope and air source heat pump',
        'description': 'Scenario 3 envelope and air source heat pump',
        'interventions': [
            'WALL_INSULATION',
            'loft_insulation',
            'double_glazing',
            'heat_pump_upgrade',
        ],
        'includes_wall_insulation': True,
        'installation_approach': 'simultaneous'
    },

    'scenario5': {
        'name': 'envelope and air heat pump, solar',
        'description': 'scenario 5: envelope, heat pump and solar',
        'interventions': [
            'WALL_INSULATION',
            'loft_insulation',
            'double_glazing',
            'heat_pump_upgrade',
            'solar_pv', 
        ],
        'includes_wall_insulation': True,
        'installation_approach': 'simultaneous'
    },

    'deep_retrofit_grouped': {
        'name': 'Deep Retrofit (Net Zero Ready)',
        'description': 'Maximum intervention package',
        'interventions': [
            'deep_retrofit_estimate',
        ],
        'includes_wall_insulation': True,
        'installation_approach': 'simultaneous'
    }
}
    