# msc/tools/user_interaction.py

def user_confirmation_tool(message: str, default: bool = True) -> str:
    """
    A tool to ask the user for confirmation with more robust handling.
    
    Args:
        message (str): The message to display to the user
        default (bool): The default choice if user just presses Enter
    
    Returns:
        str: The user's response ("y", "yes", "n", "no", or custom feedback)
    """
    try:
        default_text = "[Y/n]" if default else "[y/N]"
        response = input(f"❓ {message} {default_text}: ").strip()
        
        if not response:
            return "yes" if default else "no"
        return response.lower()
        
    except (KeyboardInterrupt, EOFError):
        print("\n⚠️  Operation cancelled by user")
        return "no"
    except Exception as e:
        print(f"❌ Error in confirmation: {e}")
        print("⚠️  Please provide your response manually:")
        try:
            response = input("Your response: ").strip()
            return response if response else ("yes" if default else "no")
        except Exception:
            return "no"

def user_feedback_tool(message: str, allow_empty: bool = False) -> str:
    """
    A tool to get detailed feedback from the user.
    
    Args:
        message (str): The message to display to the user
        allow_empty (bool): Whether to allow empty responses
    
    Returns:
        str: The user's detailed feedback
    """
    try:
        response = input(f"💭 {message}: ").strip()
        if not response and not allow_empty:
            print("⚠️  Please provide a response:")
            response = input("Your feedback: ").strip()
        return response
    except (KeyboardInterrupt, EOFError):
        print("\n⚠️  Operation cancelled by user")
        return "cancelled"
    except Exception as e:
        print(f"❌ Error getting feedback: {e}")
        return "error"