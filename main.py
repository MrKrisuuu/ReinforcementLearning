from model import DQNAgent, QAgent, device
from utils import train_agent, test_agent, get_data, get_avg, get_decays, plot_rewards, plot_diff


if __name__ == "__main__":
    epochs = 100000

    # q_agent_r01 = QAgent(r=0.1, name="r01")
    # train_agent(q_agent_r01, epochs)
    #
    # q_agent_r02 = QAgent(r=0.2, name="r02")
    # train_agent(q_agent_r02, epochs)
    #
    # q_agent_r03 = QAgent(r=0.3, name="r03")
    # train_agent(q_agent_r03, epochs)

    # q_agent_r025_lr005 = QAgent(r=0.25, lr=0.05, name="r025_lr005")
    # train_agent(q_agent_r025_lr005, epochs)

    # q_agent_r025_lr01 = QAgent(r=0.25, lr=0.1, name="r025_lr01")
    # train_agent(q_agent_r025_lr01, epochs)

    # q_agent_r025_lr02 = QAgent(r=0.25, lr=0.2, name="r025_lr02")
    # train_agent(q_agent_r025_lr02, epochs)

    # q_agent_r025_lr005_y01 = QAgent(r=0.25, lr=0.05, y=0.1, name="r025_lr005_y01")
    # train_agent(q_agent_r025_lr005_y01, epochs)
    #
    # q_agent_r025_lr005_y05 = QAgent(r=0.25, lr=0.05, y=0.5, name="r025_lr005_y05")
    # train_agent(q_agent_r025_lr005_y05, epochs)
    #
    # q_agent_r025_lr005_y09 = QAgent(r=0.25, lr=0.05, y=0.9, name="r025_lr005_y09")
    # train_agent(q_agent_r025_lr005_y09, epochs)

    # q_main = QAgent(r=0.25, lr=0.05, y=0.1, name="main")
    # train_agent(q_main, epochs)

    # q_agent_main_notuniform_15 = QAgent(r=0.25, lr=0.05, y=0.1, u=1.5, name="notuniform_15")
    # train_agent(q_agent_main_notuniform_15, epochs)
    #
    # q_agent_main_notuniform_20 = QAgent(r=0.25, lr=0.05, y=0.1, u=2, name="notuniform_20")
    # train_agent(q_agent_main_notuniform_20, epochs)

    # q_agent_main_decay_9999 = QAgent(r=0.25, lr=0.05, y=0.1, d=0.9999, name="decay_9999")
    # train_agent(q_agent_main_decay_9999, epochs)
    #
    # q_agent_main_decay_9998 = QAgent(r=0.25, lr=0.05, y=0.1, d=0.9998, name="decay_9998")
    # train_agent(q_agent_main_decay_9998, epochs)
    #
    # q_agent_main_decay_9997 = QAgent(r=0.25, lr=0.05, y=0.1, d=0.9997, name="decay_9997")
    # train_agent(q_agent_main_decay_9997, epochs)

    # q_agent_final = QAgent(r=0.25, lr=0.05, y=0.1, d=0.9997, name="final")
    # train_agent(q_agent_final, epochs)

    dqn_agent = DQNAgent(8, [100], 4, y=0.1, d=0.9997).to(device)
    # train_agent(dqn_agent, epochs)
    dqn_agent.load("final")
    test_agent(dqn_agent)

    rewards_r01 = get_data("rewards_Q_r01.txt")
    rewards_r02 = get_data("rewards_Q_r02.txt")
    rewards_r03 = get_data("rewards_Q_r03.txt")

    sizes_r01 = get_data("sizes_Q_r01.txt")
    sizes_r02 = get_data("sizes_Q_r02.txt")
    sizes_r03 = get_data("sizes_Q_r03.txt")

    rewards_r025_lr005 = get_data("rewards_Q_r025_lr005.txt")
    rewards_r025_lr01 = get_data("rewards_Q_r025_lr01.txt")
    rewards_r025_lr02 = get_data("rewards_Q_r025_lr02.txt")

    sizes_r025_lr005 = get_data("sizes_Q_r025_lr005.txt")
    sizes_r025_lr01 = get_data("sizes_Q_r025_lr01.txt")
    sizes_r025_lr02 = get_data("sizes_Q_r025_lr02.txt")

    rewards_r025_lr005_y01 = get_data("rewards_Q_r025_lr005_y01.txt")
    rewards_r025_lr005_y05 = get_data("rewards_Q_r025_lr005_y05.txt")
    rewards_r025_lr005_y09 = get_data("rewards_Q_r025_lr005_y09.txt")

    sizes_r025_lr005_y01 = get_data("sizes_Q_r025_lr005_y01.txt")
    sizes_r025_lr005_y05 = get_data("sizes_Q_r025_lr005_y05.txt")
    sizes_r025_lr005_y09 = get_data("sizes_Q_r025_lr005_y09.txt")

    rewards_main = get_data("rewards_Q_main.txt")
    rewards_notuniform_15 = get_data("rewards_Q_notuniform_15.txt")
    rewards_notuniform_20 = get_data("rewards_Q_notuniform_20.txt")

    sizes_main = get_data("sizes_Q_main.txt")
    sizes_notuniform_15 = get_data("sizes_Q_notuniform_15.txt")
    sizes_notuniform_20 = get_data("sizes_Q_notuniform_20.txt")

    rewards_decay_9999 = get_data("rewards_Q_decay_9999.txt")
    rewards_decay_9998 = get_data("rewards_Q_decay_9998.txt")
    rewards_decay_9997 = get_data("rewards_Q_decay_9997.txt")

    sizes_decay_9999 = get_data("sizes_Q_decay_9999.txt")
    sizes_decay_9998 = get_data("sizes_Q_decay_9998.txt")
    sizes_decay_9997 = get_data("sizes_Q_decay_9997.txt")

    rewards_final = get_data("rewards_Q_final.txt")
    rewards_dqn = get_data("rewards_DQN_agent.txt")

    length = 10000

    # plot_diff([get_avg(rewards_r01, length), get_avg(rewards_r02, length), get_avg(rewards_r03, length)], ["Q r=0.1", "Q r=0.2", "Q r=0.3"], title="Rewards")
    # plot_diff([sizes_r01, sizes_r02, sizes_r03], ["Q r=0.1", "Q r=0.2", "Q r=0.3"], ylabel="Size", title="Sizes")
    #
    # plot_diff([get_avg(rewards_r025_lr005, length), get_avg(rewards_r025_lr01, length), get_avg(rewards_r025_lr02, length)], ["Q lr=0.05", "Q lr=0.1", "Q lr=0.2"], title="Rewards for r=0.25")
    # plot_diff([sizes_r025_lr005, sizes_r025_lr01, sizes_r025_lr02], ["Q lr=0.05", "Q lr=0.1", "Q lr=0.2"], ylabel="Size", title="Sizes for r=0.25")
    #
    # plot_diff([get_avg(rewards_r025_lr005_y01, length), get_avg(rewards_r025_lr005_y05, length), get_avg(rewards_r025_lr005_y09, length)], ["Q y=0.1", "Q y=0.5", "Q y=0.9"], title="Rewards for r=0.25 and lr=0.05")
    # plot_diff([sizes_r025_lr005_y01, sizes_r025_lr005_y05, sizes_r025_lr005_y09], ["Q y=0.1", "Q y=0.5", "Q y=0.9"], ylabel="Size", title="Sizes for r=0.25 and lr=0.05")
    #
    # plot_diff([get_avg(rewards_main, length), get_avg(rewards_notuniform_15, length), get_avg(rewards_notuniform_20, length)], ["Q u=1.0", "Q u=1.5", "Q u=2.0"], title="Rewards for r=0.25, lr=0.05 and y=0.1")
    # plot_diff([sizes_main, sizes_notuniform_15, sizes_notuniform_20], ["Q u=1.0", "Q u=1.5", "Q u=2.0"], ylabel="Size", title="Sizes for r=0.25, lr=0.05 and y=0.1")
    #
    # plot_diff([get_avg(rewards_main, length), get_avg(rewards_decay_9999, length), get_avg(rewards_decay_9998, length), get_avg(rewards_decay_9997, length)], ["Main", "Q d=0.9999", "Q d=0.9998", "Q d=0.9997"], title="Rewards for r=0.25, lr=0.05 and y=0.1")
    # plot_diff([sizes_decay_9999, sizes_decay_9998, sizes_decay_9997], ["Q d=0.9999", "Q d=0.9998", "Q d=0.9997"], ylabel="Size", title="Sizes for r=0.25, lr=0.05 and y=0.1")
    # plot_diff([get_decays(0.9999, epochs), get_decays(0.9998, epochs), get_decays(0.9997, epochs)], ["d=0.9999", "d=0.9998", "d=0.9997"], title="Decays")

    plot_diff([get_avg(rewards_final, length), get_avg(rewards_dqn, length)], ["Q", "DQN"], title="Rewards for (r=0.25, lr=0.05 for Q), y=0.1 and d=0.9997")

