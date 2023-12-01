Example: City Scenes vs Nature Scenes
-------------------------------------

In-Context Prompt:

[{'input': 'The bustling city streets echoed with the sounds of honking horns '
           'and hurried footsteps.',
  'label': True},
 {'input': 'The cityscape was a mix of modern architecture and historic '
           'landmarks, telling the story of the urban evolution.',
  'label': True},
 {'input': 'The Louvre Museum houses famous works of art.', 'label': True},
 {'input': 'THE OCEAN WAVES CRASH AGAINST THE SHORE.', 'label': False},
 {'input': 'The Amazon River is the second-longest river in the world.',
  'label': False},
 {'input': 'mountain peaks towering above the clouds', 'label': False}]



 User Prompt Prefix:

"Please label the following inputs. Respond in JSON format like the examples given to you above."

User Prompt, Test Set:

[{'input': 'THE MOUNTAINS ARE MAJESTIC AND GRAND.'},
 {'input': 'The Taj Mahal is a symbol of love.'},
 {'input': 'Paris is known as the City of Love.'},
 {'input': "As the sun set, the city's skyline transformed into a mesmerizing "
           'display of colors and shadows.'},
 {'input': 'The Vatican City is the smallest country in the world.'},
 {'input': 'WHIRLING THROUGH THE DENSE FOG'},
 {'input': 'The Serengeti National Park is home to diverse wildlife.'},
 ...
]



