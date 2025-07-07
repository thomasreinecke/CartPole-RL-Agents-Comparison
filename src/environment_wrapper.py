# src/environment_wrapper.py (Corrected Version)

import gymnasium as gym
import pygame

class RenderInfoWrapper(gym.Wrapper):
    """
    This wrapper adds custom text rendering to the environment's UI by
    drawing on top of the frame after each step.
    """
    def __init__(self, env):
        super().__init__(env)
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 24)
        self.fall_count = 0
        self.current_episode_steps = 0

    def step(self, action):
        """
        We intercept step to check for terminations and update our counters.
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.current_episode_steps += 1
        if terminated:
            self.fall_count += 1
        return next_state, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        We intercept reset to clear the per-episode step counter.
        """
        result = self.env.reset(**kwargs)
        self.current_episode_steps = 0
        return result

    def render(self):
        """
        This is our explicit render function. It ONLY draws the text overlay.
        The underlying environment's render is called automatically by step().
        """
        # Get the underlying pygame screen surface
        screen = self.env.screen

        # Create the text surfaces
        falls_text = self.font.render(f"Falls: {self.fall_count}", True, (0, 0, 0)) # Black
        steps_text = self.font.render(f"Steps: {self.current_episode_steps}", True, (0, 0, 0))

        # Draw the text on the screen
        screen.blit(falls_text, (10, 10))
        screen.blit(steps_text, (10, 40))

        # Update the full display to show both the environment and our text
        pygame.display.flip()