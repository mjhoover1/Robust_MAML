# For supervised Learning
# Require: P(T): distribution over tasks
# Require: a, b: step size hyperparams.
# randomly initialize theta
# while not done do
    # sample batch of tasks Ti ~ p(T)
    # for all Ti do
        # Sample K datapoints D = {x_j, y_j} from Ti
        # Evaluate grad_theta L_Ti(f_theta) using D and L_Ti in Equation (2) or (3)
        # Compute adapted params with grad descent: theta_i' = theta - a*grad_theta(L_Ti(f_theta))
        # Sample datapoints Di' = {x_j, y_j} from Ti for the meta update 
    # end for 
    # Update theta <- theta - b(grad_theta(sum_Ti~p(T)L_Ti(f_theta_i'))) using each Di' and L_Ti in Equation 2 or 3
# end while

if __name__ == '__main__':
    print("hi")