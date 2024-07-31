function prettifyPassage(passage){
    let passageChunks = passage.split(`/\[\d+:\d+\]/`);
    let resultPassage = []
    for (let passage of passageChunks) {
        resultPassage.push(`<p class="passage">${passage}</p><br>`)
    }
    return resultPassage.join('\n')
}
function createHTMLFromResponse(results){
    let htmlResponse = `        
    <div class="agent-interaction">
        <div class="agent-profile">
            <img src="static/images/openArtPriest.png" alt="Chatbot's profile picture" class = "agent-pic">
            <p><b>AI</b></p>
        </div>`;
    
    let status = JSON.parse(results.status);
    console.log(status)
    if (status == 200){
        let docs = JSON.parse(results.relevant_documents);
        docs = Array.from(docs);
        if (docs.length) {
            htmlResponse += `
            <div class="agent-message">
                <h3>I've found some verses that could match what you're looking for:</p>  
            `
            for (let i = 0; i < docs.length; i++) {
                console.log(docs[i])
                console.log(typeof docs[i])
                htmlResponse += `
                    <h4>Passage number: ${i+1}</h4>
                    <h4><i>${docs[i].book}, ${docs[i].chapter}</i></h4>
                    ${prettifyPassage(docs[i].passage)}
                    `            
            }
        }
    } else if (status == 201) {
        console.log('NOT FOUND')
        htmlResponse += `
        <div class="agent-message">
            <p>I'm sorry, it seems I couldn't find relevant verses. Please try with a different question.</p>  
        `
    } else {
     htmlResponse += `
            <div class="agent-message">
                <p>I'm sorry, it seems the server is not responding</p>  
            </div>
        `
    }
    htmlResponse += `
    </div>
    `
    return htmlResponse;
    

}
async function displayNewUserMessage(message,chatHistory,userInputBar) {
    displayingMessage = true; //Disable chat while waiting for api result
    if (message.value.length) {
        await new Promise(resolve => setTimeout(resolve,50));
        chatHistory.innerHTML += `
        <div class="user-interaction">
            <div class="user-message">
                <p>${message.value}</p>
            </div>
        </div> 
        `

        let messageForAPI = message.value;
        userInputBar.value = 'Type a message here'; //Reset message
        // userInputBar.blur(); //Unfocus element
        // console.log('Message inside Func',displayingMessage);
        chatHistory.innerHTML += `
        <div class="agent-interaction" id="loading-child">
            <div class="agent-profile">
                <img src="static/images/openArtPriest.png" alt="Chatbot's profile picture" class = "agent-pic">
                <p><b>AI</b></p>
            </div>
            <div class="agent-message">
                <div class = "loader"></div>
            </div>
        </div>
        ` 
        let results = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                "Content-Type": "application/json",
              },
            body: JSON.stringify({message:messageForAPI})
        });
        results = await results.json()
        // chatHistory.innerHTML += createHTMLFromResponse(results);

        // results = await results.json();
        // Remove loader and add message
        // await new Promise(resolve => setTimeout(resolve,5000));
        let loadingElem = document.getElementById('loading-child');
        // console.log(chatHistory.children)
        await chatHistory.removeChild(loadingElem);
        // console.log(chatHistory.children)
        chatHistory.innerHTML += createHTMLFromResponse(results);
        // console.log(chatHistory.innerHTML);
        // chatHistory.innerHTML += `
        // <div class="agent-interaction">
        //     <div class="agent-profile">
        //         <img src="static/images/openArtPriest.png" alt="Chatbot's profile picture" class = "agent-pic">
        //         <p><b>AI</b></p>
        //     </div>
        //     <div class="agent-message">
        //         <p>${results.message}</p>  
        //     </div>
        // </div>
        // `
        displayingMessage = false;
    }
}
let displayingMessage = false; //Outside window event listener, otherwise it gets redifined as false!
window.addEventListener('load', () => {
    const userInputBar = document.querySelector('.user-input-bar');
    const sendBtn = document.querySelector('.send-btn');
    let chatHistorySection = document.querySelector('.chat-history-section');
        userInputBar.addEventListener('focus', ()=> {
            if (userInputBar.value == 'Type a message here') {
                userInputBar.value = '';
            }
        });
        userInputBar.addEventListener('blur', ()=> {
            if (userInputBar.value == '') {
                userInputBar.value = 'Type a message here';
            }
        });
        // User input
        sendBtn.addEventListener('click',()=>
        {   
            if ((userInputBar.value != 'Type a message here') && (!displayingMessage)){
                displayNewUserMessage(userInputBar,chatHistorySection,userInputBar);
            }
        });
        window.addEventListener('keydown',(e)=>
        {
            if ((e.key == 'Enter') && (userInputBar == document.activeElement) && (!displayingMessage)) {
                displayNewUserMessage(userInputBar,chatHistorySection,userInputBar);
            } else if ((userInputBar == document.activeElement) && (userInputBar.value == 'Type a message here') && (e.key != 'Enter')) {
                userInputBar.value = '';
            }
        });
});
// window.addEventListener('keydown', (e) => {
//     console.log(e.key)
// })