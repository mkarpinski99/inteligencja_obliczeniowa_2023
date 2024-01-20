import argparse
import gym
from collections import deque
from CarRacingDDQNAgent import CarRacingDDQNAgent
from common_functions import process_state_image
from common_functions import generate_state_frame_stack_from_queue

RENDER                        = True
STARTING_EPISODE              = 1
ENDING_EPISODE                = 500
SKIP_FRAMES                   = 2
TRAINING_BATCH_SIZE           = 64
SAVE_TRAINING_FREQUENCY       = 5
UPDATE_TARGET_MODEL_FREQUENCY = 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trenowanie Agenta DDQN do Car Racing')
    parser.add_argument('-m', '--model', help='Ścieżka do modelu, od którego zacząć uczenie')
    parser.add_argument('-s', '--start', type=int, help='Startowy epizod')
    parser.add_argument('-e', '--end', type=int, help='Końcowy epizod. Domyślnie 500')
    parser.add_argument('-p', '--epsilon', type=float, default=1.0, help='Startowy epsilon. Domyślnie 1.0')
    args = parser.parse_args()

    env = gym.make('CarRacing-v2', continuous=False)
    agent = CarRacingDDQNAgent(epsilon=args.epsilon)
    if args.model:
        agent.load(args.model)
    if args.start:
        STARTING_EPISODE = args.start
    if args.end:
        ENDING_EPISODE = args.end

    for e in range(STARTING_EPISODE, ENDING_EPISODE+1):
        init_state, _ = env.reset()
        init_state = process_state_image(init_state)

        total_reward = 0
        negative_reward_counter = 0
        state_frame_stack_queue = deque([init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        done = False
        
        while True:
            if RENDER:
                env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)

            reward = 0
            # Pomijamy n klatek co każdą klatkę gry
            for _ in range(SKIP_FRAMES+1):
                next_state, r, term, trunc, info = env.step(action)
                reward += r
                if term or trunc:
                    break

            # Zliczanie ciągłych negatywnych ocen, jeżeli minęło już 100 klatek
            negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > 100 and reward < 0 else 0

            # Nagroda za jazdę do przodu
            if action == 3:
                reward *= 1.5

            total_reward += reward

            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            done = term & trunc

            # Zapamiętywanie stanu n, akcji An i stanu n+1
            agent.memorize(current_state_frame_stack, action, reward, next_state_frame_stack, done)

            # Jeśli skończone lub 25 klatek negatywny wynik z rzędu to koniec epizodu
            if done or negative_reward_counter >= 25 or total_reward < 0:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards(adjusted): {:.2}, Epsilon: {:.2}'.format(e, ENDING_EPISODE, time_frame_counter, float(total_reward), float(agent.epsilon)))
                break
            if len(agent.memory) > TRAINING_BATCH_SIZE:
                agent.replay(TRAINING_BATCH_SIZE)
            time_frame_counter += 1


        if e % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            agent.update_target_model()

        if e % SAVE_TRAINING_FREQUENCY == 0:
            agent.save('./model/car_racingDDQN_{}.h5'.format(e))
        

    env.close()