Run the min-max agent by using the commend:
>> python ./2048.py --agent MinmaxAgent
*****
Comments:
Description of the better evaluation function:

We have created six classifying parameters:
(the log or the squared root function has been applied on most of them, in order to normalize them)

log_score_per_tile -
This corresponds to the amount of fused cells we have fused since the beginning of the game.
This value promotes higher score, and more fusion between cells.

log_max_tile -
This corresponds to the value of the tile with the highest score.
This value promotes having higher maximum tile.

num_empty_tiles -
This corresponds to the number of vacant tiles.
This value promotes doing more fusions if possible.

neighbors_score -
This number is being calculated, by going over each second tile, and checking if it has neighbors
with the exact same value, if it does then we raise this score.
This value promotes leaving the board at a good state for next moves,
because it will be easy to fuse when we get to a state with high score in this parameter.

corner_value -
This corresponds to the tiles with high numbers which are closer to the corner (top-left).
This value promotes higher value tiles staying together near the corner,
this makes it easier and more likely to fuse the high numbers with high numbers,
and the lower with the low ones.

monotonicity_penalty -
We have created a negative heuristic, that tries to encourage the tiles to be in monotonic order.
that way together with corner heuristic it tries to stay ordered near the corner.
This makes it very accessible to combine the higher tiles, which than makes it easy the higher tiles and so forth.

We have taken all of the above parameters and used weights to decide how much each value
will affect the result of the heuristic.
If we had more time, we would really be glad to try and write a genetic algorithm, in order to
decide the best combination of weights, since choosing the weights is prefect suitable for the genetic algorithm.


