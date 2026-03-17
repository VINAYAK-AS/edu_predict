// static/js/modal.js

function openModal(collegeCode, collegeName) {
    // 1. Show the popup window
    const modal = document.getElementById('college-modal');
    modal.classList.remove('hidden');

    // 2. Set the title and reset the UI to "Loading"
    document.getElementById('modal-college-name').innerText = collegeName;
    document.getElementById('modal-loader').classList.remove('hidden');
    document.getElementById('modal-data').classList.add('hidden');

    // 3. Talk to our Python API in the background!
    fetch(`/api/college-details/${collegeCode}`)
        .then(response => response.json())
        .then(data => {
            // 4. Inject the AI data into the HTML
            document.getElementById('modal-courses').innerText = data.courses;
            document.getElementById('modal-fees').innerText = data.fees;
            document.getElementById('modal-reviews').innerText = data.reviews;

            // 5. Hide the loading spinner and show the data
            document.getElementById('modal-loader').classList.add('hidden');
            document.getElementById('modal-data').classList.remove('hidden');
        })
        .catch(error => {
            console.error("Error fetching data:", error);
            document.getElementById('modal-courses').innerText = "Data unavailable.";
            document.getElementById('modal-fees').innerText = "Data unavailable.";
            document.getElementById('modal-reviews').innerText = "Could not reach AI server.";
            
            document.getElementById('modal-loader').classList.add('hidden');
            document.getElementById('modal-data').classList.remove('hidden');
        });
}

function closeModal() {
    // Hide the popup
    document.getElementById('college-modal').classList.add('hidden');
}

// Close the popup if the user clicks the dark background outside the box
window.onclick = function(event) {
    const modal = document.getElementById('college-modal');
    if (event.target == modal) {
        closeModal();
    }
}