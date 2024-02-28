# from params import params_random as params
from params import params_custom as params
from util_generate import Generator

def main(args):

    generator = Generator()
    generator.generate(args)


if __name__ == "__main__":

    main(params)
