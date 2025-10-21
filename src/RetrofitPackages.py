

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

    'scenA_add': {
        'name': 'wall_and_loft',
        'description': 'combine using multiplicative ',
        'interventions': [
            'joint_loft_wall_add',

        ],
        'includes_wall_insulation': True,
        'installation_approach': 'simultaneous'
    },
    'scenA_decay': {
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
        'name': 'heat pump and insulation',
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
    