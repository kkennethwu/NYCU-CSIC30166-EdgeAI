



# plot the score as scatter plot, x-axis is nbits, y-axis is group_size, color is score
import matplotlib.pyplot as plt


def plot_score(results):
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # plot the score as scatter plot, x-axis is nbits, y-axis is group_size, color is score
    axs[0, 0].scatter(results[:, 0], results[:, 1], c=results[:, 4])

    # plot the score as scatter plot, x-axis is nbits, y-axis is group_size, color is score
    axs[0, 1].scatter(results[:, 0], results[:, 1], c=results[:, 4])

    # plot the score as scatter plot, x-axis is nbits, y-axis is group_size, color is score
    axs[1, 0].scatter(results[:, 0], results[:, 1], c=results[:, 4])

    # plot the score as scatter plot, x-axis is nbits, y-axis is group_size, color is score
    axs[1, 1].scatter(results[:, 0], results[:, 1], c=results[:, 4])

    plt.savefig('score.png')