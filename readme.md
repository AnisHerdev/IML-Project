# Rock Paper Scissors Gesture Recognition

This project is a Rock Paper Scissors gesture recognition system using MediaPipe for hand tracking.

## Project Structure

- `testingMediapipe.py`: Main script for detecting hand gestures and classifying them as rock, paper, or scissors.
- `requirements.txt`: List of required Python packages.

## Setup

1. **Clone the repository:**

    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create a virtual environment:**

    ```sh
    python -m venv my_project_env
    ```

3. **Activate the virtual environment:**

    - On Windows:

        ```sh
        my_project_env\Scripts\activate
        ```

    - On macOS/Linux:

        ```sh
        source my_project_env/bin/activate
        ```

4. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

## Running the Project

To run the hand gesture recognition script, execute the following command:

```sh
python testingMediapipe.py
```

## How It Works

1. The script uses MediaPipe to detect hand landmarks from the webcam feed.
2. The `classify_hand_shape` function classifies the detected hand shape as rock, paper, or scissors based on the distances between specific hand landmarks.
3. The classified hand shape is displayed on the webcam feed.

## Requirements

- Python 3.6+
- OpenCV
- MediaPipe

## License

This project is licensed under the MIT License.