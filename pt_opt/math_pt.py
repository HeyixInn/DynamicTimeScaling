from typing import List
import re
from .abst_pt import AbstPt
from .abst_pt import INST_POSTION


class CODE_GEN_PT(AbstPt):
    def __init__(
        self, instruction, data_template, name, position,
    ):
        super().__init__(instruction, data_template, name, position)

    def task2pt(self, task) -> str:
        if self.position == 'PREVIOUS':
            position_word = "below"
        else:
            position_word = "above"
        keys = re.findall(r'\{([^}]+)\}', self.instruction)
        map_dict = {
            "src_lang": str(task.src_lang),
            "tgt_lang": str(task.tgt_lang),
            "position_word": str(position_word)
        }
        for k in keys:
            if k not in map_dict.keys():
                map_dict[k] = k
        try:
            instruction_str = self.instruction.format(**map_dict)
        except:
            instruction_str = self.instruction

        data_str = self.data_template.format(
            prefix=str(task.prefix),
            src_lang=str(task.src_lang),
            tgt_lang=str(task.tgt_lang),
            suffix=str(task.suffix)
        )
        if self.position == INST_POSTION.PREVIOUS:
            pt = instruction_str + '\n\n' + data_str
        else:
            pt = data_str + '\n\n' + instruction_str
        return pt



class MATH_SOLVE_PT(AbstPt):
    def __init__(
        self, instruction, data_template, name, position,
    ):
        super().__init__(instruction, data_template, name, position)


    def task2pt(self, task) -> str:
        data_str = self.data_template.format(
            question=str(task["question"]),
        )
        if self.position == INST_POSTION.PREVIOUS:
            pt = self.instruction + '\n\n' + data_str
        else:
            pt = data_str + '\n\n' + self.instruction
        return pt



# TODO
aime_pt_1 = MATH_SOLVE_PT(
        instruction="Given the following math problem, "
                    "please help to solve the following math problem for me, "
                    "and provide a boxed final answer at the end.\n\n",
        data_template="Math Problem:\n```\n{question}\n```",
        name="aime_pt_1",
        position=INST_POSTION.PREVIOUS,
)

aime_pt_2 = MATH_SOLVE_PT(
        instruction="You are a helpful mathematical assistant. "
                    "Please help me solve the following math problem, "
                    "and provide a boxed final answer at the end.\n\n",
        data_template="Math Problem:\n```\n{question}\n```",
        name="aime_pt_2",
        position=INST_POSTION.PREVIOUS,
)

aime_pt_3 = MATH_SOLVE_PT(
    instruction="Solve the following AIME problem step-by-step, "
                "then verify your solution by plugging your answer back into the original problem. "
                "Please provide a boxed final answer at the end.\n\n",
        data_template="Math Problem:\n```\n{question}\n```",
        name="aime_pt_3",
        position=INST_POSTION.PREVIOUS,
)

aime_pt_4 = MATH_SOLVE_PT(
    instruction="You can use arithmetic or algebraic computations to solve the following AIME problem. "
                "Show each calculation step, simplify expressions, "
                "and provide a boxed final answer at the end.\n\n",
    data_template="Math Problem:\n```\n{question}\n```",
    name="aime_pt_4",
    position=INST_POSTION.PREVIOUS,
)

aime_pt_5 = MATH_SOLVE_PT(
    instruction="Solve the following AIME problem step-by-step "
                "and explain your reasoning. "
                "Then provide a boxed final answer at the end.\n\n",
    data_template="Math Problem:\n```\n{question}\n```",
    name="aime_pt_5",
    position=INST_POSTION.PREVIOUS,
)

AIME_PT_LIST = [aime_pt_1, aime_pt_2, aime_pt_3, aime_pt_4, aime_pt_5]
