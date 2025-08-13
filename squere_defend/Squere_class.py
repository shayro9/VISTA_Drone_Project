import numpy as np


def p_selection(p_init, it, n_iters):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p


class SquareAttackLinfIterative:
    def __init__(self, x, p_init=0.8, maximize=True, total_iters=2000):
        """
        Initialize the iterative Linf square attack (label-agnostic, loss-driven).
        Args:
            x: Input images (N, C, H, W)
            eps: Linf perturbation bound
            p_init: Initial probability for patch size
            maximize: If True, maximize the loss; if False, minimize the loss
        """
        self.eps = 1.0
        self.x = self.initialize_stripes(x.shape)
        self.x_best = self.x.copy()
        self.p_init = p_init
        self.maximize = maximize
        self.min_val, self.max_val = 0, 1 if x.max() <= 1 else 255
        self.c, self.h, self.w = x.shape[1:]
        self.n_features = self.c * self.h * self.w
        self.best_loss = None  # Will be set after first iterate
        self.n_queries = 0
        self.total_iters = total_iters

    def initialize_stripes(self, shape):
        """
        Initialize the noise with random vertical stripes: each column is randomly -eps or +eps,
        and the value is the same for all pixels in that column (for each image and channel).
        """
        N, C, H, W = shape
        stripes = np.random.choice([-self.eps, self.eps], size=(N, C, 1, W))
        init_delta = np.tile(stripes, (1, 1, H, 1))
        return init_delta

    def apply_random_rgb_square(self) -> np.ndarray:
        """
        Overlays a randomly colored RGB square on an image, with adaptive size and value range.

        Args:
            image: np.ndarray of shape (3, H, W)
            i_iter: current iteration
            n_iters: total number of iterations
            p_init: initial pixel fraction (for square sizing)
            min_val: minimum RGB value
            max_val: maximum RGB value

        Returns:
            Modified image (np.ndarray of shape (3, H, W))
        """
        image = self.x[0]
        i_iter = self.n_queries
        n_iters = self.total_iters
        p_init = self.p_init
        min_val = self.min_val
        max_val = self.max_val
        assert image.shape[0] == 3, "Image must have shape (3, H, W)"
        C, H, W = image.shape
        new_image = image.copy()

        # Total number of features
        n_features = H * W

        # Determine p and square size
        p = p_selection(p_init, i_iter, n_iters)

        s = int(round(np.sqrt(p * n_features)))

        s = max(1, min(s, min(H, W)))  # Clamp square size to image bounds
        # Random top-left location
        top = np.random.randint(0, H - s + 1)
        left = np.random.randint(0, W - s + 1)

        # Random RGB color in [min_val, max_val]
        r = np.random.uniform(min_val, max_val)
        g = np.random.uniform(min_val, max_val)
        b = np.random.uniform(min_val, max_val)

        # Apply square
        new_image[0, top:top + s, left:left + s] = r
        new_image[1, top:top + s, left:left + s] = g
        new_image[2, top:top + s, left:left + s] = b

        return new_image

    def get_pertubed_image(self):
        return self.x.copy()

    def iterate(self, loss_value):
        """
        Perform one iteration of the square attack, updating the perturbation if the loss improves.
        Args:
            loss_value: Scalar loss value for the candidate perturbation (float or np.ndarray of shape (N,))
            x_candidate: Optional candidate perturbed image (N, C, H, W). If None, generate a new candidate.
        Returns:
            Tuple: (current x_best, current best_loss, current n_queries)
        """
        p = p_selection(self.p_init, self.n_queries, 400)

        if self.best_loss is None:
            self.best_loss = loss_value
#            self.n_queries += 1
#           return self.x_best, self.best_loss, self.n_queries
        elif (self.best_loss > loss_value and self.maximize == False) or (
                self.best_loss < loss_value and self.maximize == True):
            self.x_best = self.x.copy()
            self.best_loss = loss_value
        else:
            self.x = self.x_best.copy()
        self.n_queries += 1
        self.x[0] = self.apply_random_rgb_square()
        return self.x_best, self.best_loss, self.n_queries
