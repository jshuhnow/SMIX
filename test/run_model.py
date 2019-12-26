import sys
sys.path.insert(0, "smix/src/")

import argparse
import numpy as np
from smac.env import StarCraft2Env


# import whatever you need to defined and load your trained model
from util.my_model import MyModel


def test_model(model, env, num_runs=50):
    """
    :param model:
    :param env:
    :param num_runs:
    :return:
    """
    wins = []
    for i in range(num_runs):
        ############### you can design your test routines on here. ###############

        terminated = False
        model.my_init(env)

        while not terminated:
            actions = model.get_actions()
            reward, terminated, info = env.step(actions[0])
            model.post_action(reward,terminated,info, actions)

        if (i+1)%100 == 0:
            print(f"{i+1} / {num_runs}")
        ##########################################################################

        if terminated:
            win = True if info['battle_won'] else False
        wins.append(float(win))
        

    return np.average(wins)


def load_model(model_path) :
    """
    :param model_path:
    :return: the loaded model (an instance of torch.nn.Module expected)
    """

    """
    do whatever you want to load model. 
    In the end of this code block, we expect to get loaded_model = model
    """

    return MyModel(
        model_path=model_path,
        algs_yaml_path="smix/src/config/algs/smix.yaml",
        num_parallel=8
    )

def main(model_path):
    env = StarCraft2Env(map_name="8m",
                        window_size_x=1920 / 3,
                        window_size_y=1080 / 3,
                        difficulty="4"
    )

    loaded_model = load_model(model_path)
    mean_wr = test_model(loaded_model, env, num_runs=1000)

    env.close()
    return mean_wr


if __name__ == "__main__":
    # my_main()

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, help='path to your model')
    args = parser.parse_args()
    mean_wr = main(args.path)

    print("Your average winning rate is {}".format(mean_wr))


# For model selection

"""
def my_main():
    env = StarCraft2Env(map_name="8m",
                        window_size_x=1920 / 3,
                        window_size_y=1080 / 3,
                        difficulty="4",
                        state_last_action=True
    )



    "qmix_beta" : "../pymarl/src/config/algs/qmix_beta.yaml",
    "qtran" : "../pymarl/src/config/algs/qtran.yaml",
    "coma" : "../pymarl/src/config/algs/coma.yaml"


    algs_yaml_paths = [
        "smix/src/config/algs/smix.yaml",
        # "../pymarl/src/config/algs/qmix_beta.yaml",
        # "../pymarl/src/config/algs/qtran.yaml",
    ]

    model_paths = [
        "smix/results/models/smix__2019-12-25_04-46-04/10000741",
        # "../pymarl/results/models/qmix_smac_parallel__2019-12-17_01-54-27/2000438",
        # "../pymarl/results/models/qtran__2019-12-17_01-58-06/2000052",

        # "../pymarl/results/models/coma__2019-12-17_15-20-12/6000582",
        # "../pymarl/results/models/coma__2019-12-17_06-03-04/2000341",
        # "../pymarl/results/models/coma__2019-12-17_15-20-12/2000284",
        # "../pymarl/results/models/coma__2019-12-17_15-20-12/4000472",
    ]

    result = {}
    for i in range(len(model_paths)):
        model_path = model_paths[i]
        algs_yaml_path = algs_yaml_paths[i]
        loaded_model = load_model(model_path, algs_yaml_path, num_parallel=8)

        print(f"{model_path} start!")
        mean_wr = test_model(loaded_model, env, num_runs=500)
        result[model_path] = mean_wr
        print(f"winning raito - {mean_wr}")

    with open("test/result.txt", "w+") as my_out:
        import json
        my_out.write(json.dumps(result))

    env.close()
"""
