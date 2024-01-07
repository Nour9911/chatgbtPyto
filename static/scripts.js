document.getElementById('send-button').addEventListener('click', function() {
    let userInput = document.getElementById('user-input').value;

    fetch('/process', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: userInput })
    })
    .then(response => response.json())
    .then(data => {
        let chatMessages = document.getElementById('chat-messages');
        let userMessage = document.createElement('p');
        let botResponse = document.createElement('p');

        userMessage.textContent = `You: ${userInput}`;
        // Modify the line below to match the desired response format
        botResponse.textContent = `Bot: ${data.response}`;

        chatMessages.appendChild(userMessage);
        chatMessages.appendChild(botResponse);
    })
    .catch(error => console.error('Error:', error));

    document.getElementById('user-input').value = ''; // Clear input field after sending
});



        // document.getElementById('send-button').addEventListener('click', function() {
        //     let userInput = document.getElementById('user-input').value;

        //     fetch('/process', {
        //         method: 'POST',
        //         headers: {
        //             'Content-Type': 'application/json'
        //         },
        //         body: JSON.stringify({ message: userInput })
        //     })
        //     .then(response => response.json())
        //     .then(data => {
        //         let chatMessages = document.getElementById('chat-messages');
        //         let userMessage = document.createElement('p');
        //         let botResponse = document.createElement('p');

        //         userMessage.textContent = `You: ${userInput}`;
        //         botResponse.textContent = `Pyto: ${data.response}`;

        //         chatMessages.appendChild(userMessage);
        //         chatMessages.appendChild(botResponse);
        //     })
        //     .catch(error => console.error('Error:', error));

        //     document.getElementById('user-input').value = ''; // Clear input field after sending
        // });

