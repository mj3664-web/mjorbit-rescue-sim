"""Launch the scripted astronaut-rescue mission with visualization enabled."""

from rescue_sim.demo import main as demo_main


def main() -> None:
    demo_main(viewer_default=True)


if __name__ == "__main__":
    main()
