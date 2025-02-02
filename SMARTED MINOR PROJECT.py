import random

def guess_the_number(): 
    print("Welcome to Guess the Number!") 
    print("I am thinking of a number between 1 and 100.") 
    
    # Generate a random number between 1 and 100 
    number_to_guess = random.randint(1, 100) 
    number_of_attempts = 0 
    guessed_correctly = False 
    
    while not guessed_correctly: 
        # Get the player's guess 
        guess = input("Enter your guess: ") 
        
        # Ensure the input is an integer 
        try: 
            guess = int(guess) 
        except ValueError: 
            print("Please enter a valid number.") 
            continue 
        
        number_of_attempts += 1 
        
        # Compare the guess with the number 
        if guess < number_to_guess: 
            print("Too low!") 
        elif guess > number_to_guess: 
            print("Too high!") 
        else: 
            guessed_correctly = True 
            print(f"Congratulations! You've guessed the number in {number_of_attempts} attempts.") 

# Run the game 
guess_the_number()
