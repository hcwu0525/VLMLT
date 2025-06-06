
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