from mhr.alignment.models.llava_v1_5.llava.conversation import conv_templates, SeparatorStyle,Conversation
from mhr.vcd.experiments.eval.language_dict import language_dict

conv_llava_v1_mul = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)
system_origin = "A chat between a curious human and an artificial intelligence assistant. " \
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
           
system_ru = "Чат между любопытным человеком и искусственным интеллектом-помощником. " \
           "Помощник дает полезные, подробные и вежливые ответы на вопросы человека."

def llava_get_language_conv(lang):
    system_conv = f"A chat between a curious human and an artificial intelligence assistant. The chat is in {language_dict[lang]["full_name"]}." \
                    f"The assistant gives helpful, detailed, and polite answers in {language_dict[lang]["full_name"]}"
    conv = Conversation(
        system=system_conv,
        roles=("USER", "ASSISTANT"),
        version="v1",
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.TWO,
        sep=" ",
        sep2="</s>",
    )
    return conv
