
what_word_instruction=f"""
 You are a helpful AI assistant, and you are hearing a piece of conversation about one image.
 Read the conversation about one image and tell the interrogative words or ways of asking questions used in the conversation. There are several requirements to note。
 1.Be aware that your answer should only contain a list of the origin format of the questioning method of the conversation. For example, If the mentioned phrase is 'What is the color of...', you should only return 'what color' instead of 'what is the color of' or 'what be'. 
 2.The answers should be separated by commas, e.g. a,b,c,... 
 3.Make sure your answer only contains the questioning method list and does not contain any other word.
 4.The questioning method can be composed of single or multiple words.
 5.If it is an imperative sentence, state what the questioner is asking to be done.
 """
what_word_conv1=f"""
 human: Write a terse but informative summary of the picture.\n<image>
 gpt: a grey watch with an army style strap
 """
what_word_ans1="describe"
what_word_conv2="""
human:Give a brief description of the image.\n<image>
gpt:the divine queen in her elaborate masks canvas print featuring the face and hands of a woman with red hair
"""
what_word_ans2="describe"
what_word_conv3=f"""
human: Where is the cat positioned in the image?\n<image>
gpt: The cat is positioned on top of the back of the couch in the living room.
human: What is the cat doing in the image?
gpt:The cat is coming out from some curtains onto the couch and is sitting or standing on top of it.
human: What is the color of the curtains near the cat?
gpt:The color of the curtains near the cat is red.
human: Besides the cat, what else can be seen on the couch?
gpt:There are lots of pillows on the couch besides the cat.
human: What kind of room does the couch appear to be in?
gpt:The couch appears to be in a living room setting.
"""
what_word_ans3="where,what be doing,what color,what else,what kind of"

what_word_conv4=f"""
human: What type of sign and traffic device can be seen in the image?\n<image>
gpt: In the image, there is a street sign and a traffic light above a city road.
human: How many traffic lights are visible in the image?
gpt: There are a total of six traffic lights visible in the image.
human: What is the current color of the traffic lights in the image?
gpt: All the traffic lights in the image are showing the color red at the same time.
human: What kind of setting is the image taken in?
gpt: The image is taken in a big city setting with a city street, an overpass, and buildings in the distance.
human: Are there any cars visible in the image?
gpt: Yes, there are cars visible on the city street in the image.
"""
what_word_ans4="what type,how many,what color,what kind of,be there"



existence_instruction=f"""
 You are an AI visual assistant, and you are seeing a single image. What you see are provided with one piece of conversation, describing the same image you are looking at. Read the conversation about one image and tell what object(s) are really shown in the image. There are several requirements to note。
 1.Be aware that your answer should only contain a list of the origin format of the word of the object(s) in the image. For example, If the mentioned object is a 'dog', you should only return 'dog' instead of 'dogs' or 'doggy'. 
 2.The answers should be separated by commas, e.g. a,b,c,... 
 3.Make sure your answer only contains the object list and does not contain any other word.
 4.The object can be nouns or noun phrases.
 """
existence_conv1=f"""
 human: Write a terse but informative summary of the picture.\n<image>
 gpt: a grey watch with an army style strap
 """
existence_ans1="watch"
existence_conv2="""
human:Give a brief description of the image.\n<image>
gpt:the divine queen in her elaborate masks canvas print featuring the face and hands of a woman with red hair
"""
existence_ans2="divine queen,mask"
existence_conv3=f"""
human: Where is the cat positioned in the image?\n<image>
gpt: The cat is positioned on top of the back of the couch in the living room.
human: What is the cat doing in the image?
gpt:The cat is coming out from some curtains onto the couch and is sitting or standing on top of it.
human: What is the color of the curtains near the cat?
gpt:The color of the curtains near the cat is red.
human: Besides the cat, what else can be seen on the couch?
gpt:There are lots of pillows on the couch besides the cat.
human: What kind of room does the couch appear to be in?
gpt:The couch appears to be in a living room setting.
"""
existence_ans3="cat,couch,pillow,curtain,living room"

translate_instruction="""
 You are a helpful AI assistant translator. You are given a piece of sentence or a word, and you are asked to translate it from English into {}. Be aware that your answer should only contain the translation of the sentence or word. Make sure your answer is accurate and does not contain any other word.
"""

sentence_extract_object_instruction=f"""
 You are an AI visual assistant, and you are seeing a single image. What you see are provided with a few of sentences, describing the same image you are looking at. Read the sentences about one image and tell what object(s) are really shown in the image. There are several requirements to note。
 1.Be aware that your answer should only contain a list of the origin format of the word of the object(s) in the image. For example, If the mentioned object is a 'dog', you should only return 'dog' instead of 'dogs' or 'doggy'. 
 2.The answers should be separated by commas, e.g. a,b,c,... 
 3.Make sure your answer only contains the object list and does not contain any other word.
 4.The object can be nouns or noun phrases.
 """
 
sentence_extract_object_conv1=f"""
The image features a man riding a motorcycle on a street, with a car following behind him. 
The man on the motorcycle is wearing a helmet and appears to be looking back at the car. 
There are several other people in the scene, some of them standing or walking nearby.\n\n
In addition to the motorcycle and car, there is a backpack visible in the scene, possibly belonging to one of the people present. 
The overall atmosphere suggests a busy street with various individuals going about their day.
"""
sentence_extract_object_ans1 = f"""
man,motorcycle,street,car,helmet,backpack,people
"""

sentence_extract_object_conv2 = f"""
The image features a group of four horned animals, likely sheep, standing together in a grassy field. 
They are positioned in a line, with one sheep on the left side, another in the middle, and two more on the right side of the field. 
The animals appear to be looking in the same direction, possibly towards the camera.\n\n
The field is surrounded by a fence, which can be seen in the background, providing a sense of enclosure for the animals. 
The scene captures the animals in their natural environment, enjoying the open space and the company of their fellow herd members.
"""
sentence_extract_object_ans2 = f"""
sheep,field,line,fence,enclosure,animals,environment
"""

sentence_extract_object_conv3 = f"""
The image features a large airport runway with several airplanes parked on the tarmac. 
There are three airplanes in the scene, with one positioned towards the left side, another in the middle, and the third one on the right side of the runway. \n\n
In addition to the airplanes, there are several trucks scattered around the tarmac, likely providing support services to the aircraft. 
Some of these trucks are located near the left side of the runway, while others are positioned closer to the center and right side.\n\n
A person can also be seen in the scene, standing near the right side of the runway, possibly attending to the airplanes or performing maintenance tasks.
"""
sentence_extract_object_ans3 = f"""
runway,airplanes,tarmac,trucks,person
"""

llava_instruction = """
You are an AI visual assistant, and you are seeing a single image. What you see are provided with a few of sentences, describing the same image you are looking at. Answer all questions as you are seeing the image.

Design a conversation between you and a person asking about this photo. The answers should be in a tone that a visual AI assistant is seeing the image and answering the question.
Ask diverse questions and give corresponding answers.

Include questions asking about the visual content of the image, including the object types, counting the objects, object actions, object locations, relative positions between objects, etc. Only include questions that have definite answers:
(1) one can see the content in the image that the question asks about and can answer confidently;
(2) one can determine confidently from the image that it is not in the image.
Do not ask any question that cannot be answered confidently.

Also include complex questions that are relevant to the content in the image, for example, asking about background knowledge of the objects in the image, asking to discuss about events happening in the image, etc. Again, do not ask about uncertain details.
Provide detailed answers when answering complex questions. For example, give detailed examples or reasoning steps to make the content more convincing and well-organized.  You can include multiple paragraphs if necessary.
"""

llava_conv1 = """
There is a movie theater that displays the show times above the doors.
A red fire hydrant is deep in the snow.
The fire hydrant is in the snow near a recently plowed sidewalk.
This city has had a very hard winter with snow.
A hotel for dogs in the snow in winter.
"""
llava_ans1 = """
Question:
What color is the fire hydrant in the image?
===
Answer:
The fire hydrant in the image is red.
===
Question:
Is the fire hydrant fully covered by snow or partially visible?
===
Answer:
The fire hydrant is partially visible, with a significant portion of it covered by the deep snow.
===
Question:
What is the condition of the sidewalk near the fire hydrant?
===
Answer:
The sidewalk near the fire hydrant has been recently plowed, so it is relatively clear of snow compared to the surrounding area.
===
Question:
What does the image suggest about the winter season in the city?
===
Answer:
The image suggests that the winter season in the city has been quite harsh and challenging. The significant snow accumulation seen in the image indicates that the city has experienced heavy snowfall, which can lead to various difficulties and disruptions for its residents. 

For instance, the deep snow covering the fire hydrant could potentially hinder its accessibility in case of emergencies, which poses safety concerns. The recently plowed sidewalk implies that the city's maintenance crews have been actively working to keep the walkways clear and safe for pedestrians, but the sheer amount of snowfall might make it difficult to keep up with the continuous snow removal efforts. 

Furthermore, such extreme winter conditions can impact transportation, leading to traffic issues, delays in public transit, and increased risks of accidents due to slippery roads. It can also cause problems for businesses and schools, as people might face challenges commuting to work or attending classes. Additionally, the heavy snow can put extra strain on infrastructure, such as roofs and power lines, increasing the likelihood of structural damage or power outages. 

In conclusion, the image of the red fire hydrant deep in the snow and the recently plowed sidewalk suggest that the city has faced a particularly severe winter season, with substantial snowfall that has likely caused various challenges and disruptions for its residents and infrastructure.
"""

llava_conv2="""
A man is skiing in the open snow covered hills
A skier is making his way into the snow.
A skier on the foothills of a huge mountain range.
A skier looks at mountains while standing near a trail sign.
a single person skiing by an area with a lot of bushes
"""

llava_ans2="""
Question:
What is the skier doing?
===
Answer:
The skier is skiing in the snow-covered hills, making his way through the snow and enjoying the beautiful mountain scenery. They are also standing near a trail sign, which indicates their location on the mountain.
===
Question:
Are there any other people or skiers visible in the image?
===
Answer:
No, there is only a single person skiing in the image, and they are surrounded by an area with a lot of bushes.
===
Question:
What kind of skiing is the person doing? Cross-country or downhill?
===
Answer:
Based on what I can see in the photo, it is not possible to definitively determine whether the skier is doing cross-country or downhill skiing. However, given the presence of a trail sign and the fact that they are skiing in the foothills, it is more likely that the skier is engaged in cross-country skiing. Downhill skiing typically takes place on more steep slopes and involves ski lifts, whereas cross-country skiing often occurs on relatively flatter terrain with marked trails.
===
Question:
Can you explain cross-country and downhill skiing a bit more?
===
Answer:
Cross-country skiing and downhill skiing are two distinct types of skiing, each with its own techniques, equipment, and terrain preferences. 

1. Cross-Country Skiing:
Also known as Nordic skiing, cross-country skiing involves traversing relatively flat terrain or gentle slopes using skis and poles. It is a popular recreational and competitive sport that focuses on endurance, balance, and aerobic fitness. Cross-country skiers use their own locomotion to move across snow-covered terrain, rather than relying on ski lifts or other forms of assistance. 
There are two main techniques in cross-country skiing:
- Classic: This technique involves skiing in parallel tracks, using a forward-and-backward gliding motion while keeping the skis parallel to each other. The skier uses their poles to propel themselves forward.
- Skate skiing: This technique is similar to ice skating, where the skier pushes off one ski at an angle and glides on the other. It requires a smooth, groomed surface and is typically faster than the classic technique. 
Cross-country ski equipment consists of lightweight skis, boots, bindings, and poles. The boots are more flexible than those used in downhill skiing, allowing for better ankle movement and control. 

2. Downhill Skiing:
Also known as alpine skiing, downhill skiing involves descending slopes at high speeds using skis and poles for balance and control. This sport is more focused on speed, technique, and navigating challenging terrain, including steep slopes, moguls, and even jumps. 
Downhill skiing can be further categorized into several disciplines, such as slalom, giant slalom, super-G, and downhill racing. Each discipline has its own set of rules, courses, and ski equipment. 
Downhill ski equipment includes heavier and stiffer skis, boots, bindings, and poles than those used in cross-country skiing. The boots are more rigid to provide better support and control during high-speed descents and sharp turns. 

In summary, cross-country skiing is an endurance-based sport that involves traveling across flat or gently sloping terrain, while downhill skiing is focused on speed and technique as skiers navigate steeper slopes and challenging terrain. Both sports require specialized equipment and techniques, but they offer different experiences and challenges to participants.
"""

token_aug_instruction=f"""
You are an AI language assistant, and you are listening to a conversation between a person and an AI visual assistant. This conversation is about an image.

Rephrase the conversation about the image using the words given after 'Candidate words'. 
The rephrased conversation should maintain the same meaning as the original conversation. There are several requirements to note.

(1) The rephrased conversation should be coherent and grammatically correct.
(2) You can choose one or more words from the 'Candidate words' list to replace the original words in the conversation.
(3) You can use the words in the 'Candidate words' list in any form (e.g., noun, verb, adjective, etc.) for one time or multiple times in the conversation.
(4) Not all words in the 'Candidate words' list need to be used in the conversation. You can decide whether to use them based on the context.
(5) You can extend the conversation or change the number of the rounds of the conversation but make sure not to change the original meaning of the conversation.

"""

token_aug_conv1=f"""
Conversation:
[
Question: What color is the traffic light shown in the image?
<image>
Answer: The traffic light in the image is green.
Question: How does the traffic appear to be moving at the intersection?
Answer: Traffic appears to be moving smoothly through the intersection, with cars continuing on their way after the green light.
Question: Is this image taken during the day or at night?
Answer: The image is taken at night.
Question: How are the traffic lights positioned in relation to the road?
Answer: The traffic lights are suspended above the road, hanging from a pole.
Question: How do the cars look in the image due to their motion?
Answer: Due to their motion, the cars appear as streaks passing by the traffic signals in the image. This effect indicates they are moving at moderate to high speeds while the image was captured.
]
===
Candidate words:
[overwinter, wintertime, set_phrase, give_voice, musical_phrase, phrasal_idiom, idiomatic_expression, articulate, formulate, idiom, get_behind, chase_after, drop_behind, hang_back, give_chase, go_after, drop_back, shack, twelvemonth, yr, railroad_car, elevator_car, cable_car, railway_car, motorcar, railcar, gondola, promiscuous, unaccented, light_up, abstemious, Light_Within, light_source, low-cal, Inner_Light, get_down, luminosity, wakeful, sluttish, luminousness, lightheaded, light-colored, fire_up, unclouded, visible_radiation, scant, visible_light, lightly, unhorse, light-headed, get_off, calorie-free, lightsome, swooning, illume, illumine, brightness_level, Christ_Within, ignitor, alight, wanton, weak, luminance, igniter, lighter, tripping, ignite, loose, faint, dismount, idle, illuminate, sparkle, twinkle, lightness, lite, easy, look-alike, figure_of_speech, simulacrum, range_of_a_function, mental_image, visualise, ikon, visualize, envision, effigy, trope, epitome, fancy, paradigm, see, word_of_honor, Holy_Writ, give_voice, countersign, Holy_Scripture, Good_Book, watchword, tidings, Christian_Bible, Word_of_God, give-and-take, articulate, Logos, formulate, parole, Son, Scripture, oppugn, interrogative_sentence, interrogate, dubiousness, doubtfulness, call_into_question, interrogative, interrogation, enquiry, fourth_dimension, prison_term, clock_time, metre]
"""
token_aug_ans1=f"""

Question: What hue is the traffic signal displayed in the visual?
===
Answer: The traffic signal shown is illuminated in green.
===
Question: In what manner is the vehicular movement at the crossing?
===
Answer: Vehicular movement at the crossing is unobstructed, with motorcars proceeding post the green illumination.
===
Question: Was this visual captured during daylight or after dusk?
===
Answer: This visual was captured after dusk.
===
Question: In what relation are the traffic signals positioned to the roadway?
===
Answer: The traffic signals are suspended over the roadway, hanging from a pole.
===
Question: What appearance do the automobiles present in the visual due to their motion?
===
Answer: Owing to their motion, the automobiles are depicted as blurs traversing past the traffic signals, indicating their brisk pace at the time of capture.
"""
token_aug_conv2=f"""
Conversation:
[
Question: <image>
Is the train situated close to water in the image?
Answer: Yes, the image shows a small steam railroad moving quickly on a track next to water.
Question: Can you describe the surroundings of the train?
Answer: The train is surrounded by some bushes and trees, a building, and a dock by a river, providing a picturesque setting. There is also a pub and a grassy hill in the scene, which appears to be in a rainy rural town.
Question: What is the quality of the image?
Answer: The quality of the image is rather fuzzy, potentially making some details difficult to discern.
Question: Is there steam visible coming from the train as it travels?
Answer: Yes, the trains are blowing steam as they travel, which is characteristic of steam locomotives.
Question: Can you briefly explain how steam locomotives work and why they emit steam?
Answer: Steam locomotives are powered by steam engines that use heat to produce steam from water, which in turn drives the locomotive's wheels to generate motion. The key components of a steam locomotive engine include a firebox, boiler, cylinders, pistons, and drive wheels.

The process of operating a steam locomotive involves the following steps:

1. Fuel (usually coal or wood) is burned in the firebox, creating heat.
2. The heat from the firebox is transferred to the tightly sealed boiler, which is filled with water. This heat causes the water inside the boiler to boil and produce steam.
3. The steam generated in the boiler expands and is channeled to the cylinders, where it pushes the pistons back and forth. This back-and-forth motion is transferred to the drive wheels through a set of connecting rods, causing the wheels to rotate and propel the locomotive forward.
4. The used steam is then released from the cylinders through the smokestack (or chimney), which is what we see as the characteristic steam emission from the locomotive.

So, the visible steam emission from steam locomotives is a byproduct of the engine's operation. As the steam completes its work in pushing the pistons and driving the locomotive forward, it is expelled through the smokestack, contributing to the iconic steam trail associated with these classic trains.
]
===
Candidate words:
[set_phrase, give_voice, musical_phrase, phrasal_idiom, idiomatic_expression, articulate, formulate, idiom, motion_picture, word_picture, picture_show, motion-picture_show, pictorial_matter, visualise, ikon, mental_picture, moving-picture_show, word-painting, moving_picture, depict, characterisation, delineation, visualize, envision, characterization, flick, fancy, see, Harlan_F._Stone, Isidor_Feinstein_Stone, Harlan_Fisk_Stone, endocarp, I._F._Stone, Harlan_Fiske_Stone, Harlan_Stone, lapidate, Lucy_Stone, Oliver_Stone, Edward_Durell_Stone, gemstone, look-alike, figure_of_speech, simulacrum, range_of_a_function, mental_image, visualise, ikon, visualize, envision, effigy, trope, epitome, fancy, paradigm, see, word_of_honor, Holy_Writ, give_voice, countersign, Holy_Scripture, Good_Book, watchword, tidings, Christian_Bible, Word_of_God, give-and-take, articulate, Logos, formulate, parole, Son, Scripture, oppugn, interrogative_sentence, interrogate, dubiousness, doubtfulness, call_into_question, interrogative, interrogation, enquiry, ramp_up, build_up, work_up, establish, edifice, construct, colourise, people_of_color, colouring_material, colour_in, distort, color_in, colorise, coloring_material, discolour, discolor, emblazon, people_of_colour, colourize, colouration, colorize, vividness, colouring, semblance, tinge, gloss]
"""

token_aug_ans2=f"""
Question: In the visual, is the locomotive proximate to aquatic features?
===
Answer: Yes, the picture displays a diminutive steam locomotive briskly navigating tracks adjacent to a body of water.
===
Question: Could you delineate the locomotive’s environs?
===
Answer: The locomotive is ensconced among bushes and trees, an edifice, and a dock by a river, crafting a picturesque milieu. Additionally, the scene includes a public house and a verdant mound, set in a seemingly precipitous rural township.
===
Question: What is the visual quality of the image?
===
Answer: The image’s quality is notably blurry, which might obscure certain details.
===
Question: Is there steam discernible emanating from the locomotive while in transit?
===
Answer: Indeed, the locomotives emit steam while in motion, a hallmark of steam engines.
===
Question: Could you succinctly explicate the function of steam locomotives and the reason for their steam emission?
===
Answer: Steam locomotives operate on steam engines that convert heat into mechanical motion by vaporizing water to produce steam. This steam is then directed into cylinders where it moves pistons. These pistons, in turn, drive the locomotive’s wheels. The primary components of such an engine include a firebox, boiler, cylinders, pistons, and driving wheels.

The operation sequence is as follows:

	1.	Fuel, typically coal or wood, is combusted in the firebox to generate heat.
	2.	This heat boils the water within the sealed boiler, producing steam.
	3.	The steam pushes the locomotive’s pistons which, connected by rods, turn the wheels, propelling the locomotive.
	4.	Post-mechanical work, the spent steam escapes through the chimney, visible as steam emissions. This is the steam’s visible trail, a characteristic byproduct of the locomotive’s operation, marking its passage.
"""

tokenwhatword_rephrase_instruction=f"""
You are an AI language assistant, and you are listening to a conversation between a person and an AI visual assistant. This conversation is about an image.

Be creative and imaginative to rephrase the conversation about the image. 
The rephrased conversation should maintain the same meaning as the original conversation. There are several requirements to note.

(1) The rephrased conversation should be coherent and grammatically correct.
(2) You can change the words in the conversation to their synonyms or similar words, you can change the tune of speech, or you can change the sentence structure.
(3) You can extend the conversation or change the number of the rounds of the conversation but make sure not to change the original meaning of the conversation.

"""

tokenwhatword_rephrase_conv1=f"""
Question: What color is the traffic light shown in the image?
<image>
Answer: The traffic light in the image is green.
Question: How does the traffic appear to be moving at the intersection?
Answer: Traffic appears to be moving smoothly through the intersection, with cars continuing on their way after the green light.
Question: Is this image taken during the day or at night?
Answer: The image is taken at night.
Question: How are the traffic lights positioned in relation to the road?
Answer: The traffic lights are suspended above the road, hanging from a pole.
Question: How do the cars look in the image due to their motion?
Answer: Due to their motion, the cars appear as streaks passing by the traffic signals in the image. This effect indicates they are moving at moderate to high speeds while the image was captured.
"""

tokenwhatword_rephrase_ans1=f"""
Question: What color is the traffic light shown in the image?
===
Answer: The traffic light in the image is green.
===
Question: How does the traffic appear to be moving at the intersection?
===
Answer: Traffic seems to be flowing smoothly at the intersection, with vehicles continuing their journey after the green light.
===
Question: Is this image taken during the day or at night?
===
Answer: The image was taken at night.
===
Question: How are the traffic lights positioned in relation to the road?
===
Answer: The traffic lights are hanging above the road, suspended from a pole.
===
Question: How do the cars look in the image due to their motion?
===
Answer: The cars appear as streaks in the image, indicating they were moving at a moderate to high speed when the photo was taken.
"""
tokenwhatword_rephrase_conv2=f"""
Question: What is the girl eating in the image?
Answer: The girl in the image is eating a dessert, which appears to be a graham cracker treat or a cookie sandwich.
Question: Describe the girl's hair color and clothing.
Answer: The girl has blonde hair, and she is wearing a pink shirt.
Question: What color is the plate that the dessert is on?
Answer: The dessert is on a green plate.
Question: Is the girl looking at the camera or focusing on her dessert?
Answer: The girl is looking up at the camera while taking a bite of her dessert.
Question: Where is the girl eating her dessert?
Answer: The girl is eating her dessert at the table.
"""

tokenwhatword_rephrase_ans2=f"""
Question: What is the girl enjoying in the picture?
===
Answer: The girl is savoring a dessert, which looks like it could be a graham cracker treat or perhaps a cookie sandwich.
===
Question: How would you describe the girl’s hair and outfit?
===
Answer: She has blonde hair and is dressed in a pink top.
===
Question: What is the color of the plate holding the dessert?
===
Answer: The dessert is served on a green plate.
===
Question: Is the girl’s attention on the camera or her dessert?
===
Answer: She’s looking up at the camera as she takes a bite of her dessert.
===
Question: Where is the girl as she eats her dessert?
===
Answer: She’s enjoying her dessert while seated at the table.
"""

extract_replace_words_instruction=f"""
You are an AI language assistant, and you are listening to a conversation between a person and an AI visual assistant. This conversation is about an image.

Now I want to rephrase the sentence, so I need to know which words can be replaced with the words in the given list.
Which words can the words in the given list be replaced with in the context of the current conversation? Please help me list them.
There are several requirements to note.
(1) The candidate words are given after 'Candidate words'.
(2) For each word or phrase listed in 'Candidate words', you need to identify what other words can be used to replace it. If there is none, omiss the word.
(3) You can add new words to the list if you think they can be replaced in the conversation.
(4) The format of your answer should be 'word1:replacement1,replacement2,...\n===\nword2:replacement3,replacement4,...'. If there is no replacement, you can omit the word.
"""


prompt_dict={
    "existence_conversation":{
        "system":existence_instruction,
        "fewshot":[(existence_conv1,existence_ans1),(existence_conv2,existence_ans2),(existence_conv3,existence_ans3)]
    },
    "what_word_conversation":{
        "system":what_word_instruction,
        "fewshot":[(what_word_conv1,what_word_ans1),(what_word_conv2,what_word_ans2),(what_word_conv3,what_word_ans3),(what_word_conv4,what_word_ans4)]
    },
    "sentence_extract_object":{
        "system":sentence_extract_object_instruction,
        "fewshot":[(sentence_extract_object_conv1,sentence_extract_object_ans1),(sentence_extract_object_conv2,sentence_extract_object_ans2),(sentence_extract_object_conv3,sentence_extract_object_ans3)]
    },
    "llava_caption_to_conversation":{
        "system":llava_instruction,
        "fewshot":[(llava_conv1,llava_ans1),(llava_conv2,llava_ans2)]
    },
    "token_augmentation":{
        "system":token_aug_instruction,
        "fewshot":[(token_aug_conv1,token_aug_ans1),(token_aug_conv2,token_aug_ans2)]
    },
    "tokenwhatword_rephrase":{
        "system":tokenwhatword_rephrase_instruction,
        "fewshot":[(tokenwhatword_rephrase_conv1,tokenwhatword_rephrase_ans1),(tokenwhatword_rephrase_conv2,tokenwhatword_rephrase_ans2)]
    }
}

def test_deepseek():
    from openai import OpenAI
    prompt = prompt_dict["language_existence"]
    messages=[{"role": "system", "content":prompt["system"]}]
    for shot in prompt["fewshot"]:
        messages.append({"role": "user", "content": shot[0]})
        messages.append({"role": "assistant", "content": shot[1]})
    query=f"""
    human:<image>\nAre the people in the image skiing downhill or cross-country skiing?
    gpt:The people in the image are cross-country skiing in the woods, as they are skiing on a trail rather than a steep slope.
    human:How many people are in the image?
    gpt:There are two people in the image, both on skis in the snow.
    human:What kind of environment are they skiing in?
    gpt:They are skiing in a wooded environment, following a trail through the trees while surrounded by snow.
    human:Do the skiers have any additional gear with them besides their skis and poles?
    gpt:Yes, the two male skiers are carrying backpacks while they ski through the woods. The backpacks might contain essentials for their skiing adventure, such as food, water, extra clothing, or safety equipment.
    """
    messages.append({"role": "user", "content": query})


    client = OpenAI(api_key="sk-1f91217ff9824e4d914b30d81778fb86", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False,
        temperature=0,
    )

    print(response.choices[0].message.content)
    

if __name__ == "__main__":
    test_deepseek()