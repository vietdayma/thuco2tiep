from controllers.emission_controller import EmissionController
from views.main_view import MainView

def main():
    # Initialize controller
    controller = EmissionController()
    
    # Train the model
    try:
        test_score = controller.initialize_model('co2 Emissions.csv')
        print(f"Model trained successfully. Test score: {test_score:.3f}")
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return

    # Initialize and show view
    view = MainView(controller)
    view.show()

if __name__ == "__main__":
    main() 