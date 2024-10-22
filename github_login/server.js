const express = require('express');
const path = require('path');
const axios = require('axios');
const bodyParser = require('body-parser');

const app = express();
const PORT = 3000;

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Serve stype files
app.use(express.static(path.join(__dirname)));

// Serve login page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'login_page.html'));
});

// Start the login process
app.post('/start-login', (req, res) => {
    const client_id = 'Ov23liTBi9c7RB4i7SA4';  // Replace with your actual GitHub OAuth Client ID
    const redirect_uri = 'http://localhost:3000/oauth-callback';  // Ensure this matches your registered callback URL

    // Construct the GitHub OAuth URL
    const githubAuthURL = `https://github.com/login/oauth/authorize?client_id=${client_id}&redirect_uri=${redirect_uri}&scope=user`;

    // Respond with the auth URL
    res.json({ auth_url: githubAuthURL });
});

// OAuth callback
app.get('/oauth-callback', async (req, res) => {
    const code = req.query.code;  // GitHub sends the authorization code
    const client_id = 'Ov23liTBi9c7RB4i7SA4';  // Replace with your GitHub app's client ID
    const client_secret = '5e307503a552e78048d456756216b786eb4cb622';  // Replace with your GitHub app's client secret

    try {
        // Exchange the authorization code for an access token
        const response = await axios.post('https://github.com/login/oauth/access_token', null, {
            params: {
                client_id,
                client_secret,
                code
            },
            headers: {
                accept: 'application/json'
            }
        });

        // Extract the access token from GitHub's response
        const accessToken = response.data.access_token;

        if (accessToken) {
            // Fetch user info
            const userInfoResponse = await axios.get('https://api.github.com/user', {
                headers: {
                    Authorization: `token ${accessToken}`
                }
            });

            const userInfo = userInfoResponse.data;

            // Log the user information to the console
            console.log(userInfoResponse.data);

            console.log(userInfoResponse.data.login);

            const userInfoString = encodeURIComponent(JSON.stringify(userInfo));

            // After successful authentication, redirect --> main page
            res.redirect(`/main?token=${accessToken}&userInfo=${userInfoString}`);
        } else {
            res.send('Error: Access token not received.');
        }
    } catch (error) {
        console.error('Error fetching the access token:', error);
        res.send('Error during GitHub authentication');
    }
});

// Serve the main page
app.get('/main', (req, res) => {
    res.sendFile(path.join(__dirname, 'main_page.html'));  // Ensure the correct path to main_page.html
});

// Start
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
