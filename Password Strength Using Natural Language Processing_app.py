import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt

# Connect to the database
con = sqlite3.connect(r'/Users/PinnaclePulse/Innovation Environment/DataScience Project-2/password_data.sqlite')

# Load data from the database
data = pd.read_sql_query("SELECT * FROM Users", con)

# Data cleaning
data.drop(['index'], axis=1, inplace=True)
data['length'] = data['password'].str.len()

def freq_lowercase(row):
    return len([char for char in row if char.islower()]) / len(row)

data['lowercase_freq'] = np.round(data['password'].apply(freq_lowercase), 3)

def freq_special_case(row):
    special_chars = [char for char in row if not char.isalpha() and not char.isdigit()]
    return len(special_chars)

data['special_case_freq'] = data['password'].apply(freq_special_case) / data['length']

# Convert passwords to TF-IDF vectors
vectorizer = TfidfVectorizer(analyzer="char")
X = vectorizer.fit_transform(data['password'])

# Add length and lowercase frequency as features
df2 = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
df2['length'] = data['length']
df2['lowercase_freq'] = data['lowercase_freq']

# Prepare the data for training
y = data['strength']
X_train, X_test, y_train, y_test = train_test_split(df2, y, test_size=0.20)

# Train the model
clf = LogisticRegression(multi_class='multinomial')
clf.fit(X_train, y_train)

# Define the function to predict password strength
def predict_password_strength(password):
    sample_array = np.array([password])
    sample_matrix = vectorizer.transform(sample_array)
    length_password = len(password)
    length_normalised_lowercase = len([char for char in password if char.islower()]) / len(password)
    new_matrix2 = np.append(sample_matrix.toarray(), (length_password, length_normalised_lowercase)).reshape(1, 101)
    result = clf.predict(new_matrix2)
    if result == 0:
        return "Password is weak"
    elif result == 1:
        return "Password is normal"
    else:
        return "Password is strong"

# Streamlit App
st.title("Password Strength Checker using Natural Language Processing")

password_input = st.text_input("Enter a password:")

if st.button("Check Password Strength"):
    if password_input:
        strength = predict_password_strength(password_input)
        st.success(f"Password strength: {strength}")
    else:
        st.warning("Please enter a password.")

# Optionally, display model performance and graphs
if st.checkbox("Show model performance and interactive graphs"):
    y_pred = clf.predict(X_test)
    st.subheader("Model Performance")
    st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
    
    # Display confusion matrix as a Plotly heatmap
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues')
    fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig_cm)

    st.subheader("Interactive Visualizations with Professional Insights")
    
    # Password Strength Distribution
    st.write("**Password Strength Distribution**")
    fig_pie = px.pie(data, names='strength', title="Password Strength Distribution")
    st.plotly_chart(fig_pie)
    st.write("""
    **Insight**: This pie chart shows the distribution of password strength categories across the dataset. It highlights the proportion of weak, normal, and strong passwords, revealing user behavior trends.
    **Recommendation**: If a significant portion of passwords falls into the weak category, it's crucial to enhance password policies. Consider implementing stricter password requirements and educating users on the importance of password security.
    """)

    # Distribution of Password Lengths
    st.write("**Distribution of Password Lengths**")
    fig_length = px.histogram(data, x='length', nbins=20, title="Distribution of Password Lengths")
    st.plotly_chart(fig_length)
    st.write("""
    **Insight**: The histogram reveals the distribution of password lengths across the dataset. A significant concentration of shorter passwords (e.g., 6-8 characters) may suggest a common user behavior of opting for minimal effort in password creation. This poses a security risk, as shorter passwords are more susceptible to brute-force attacks. 
    **Recommendation**: Encourage users to create longer passwords (12+ characters) to improve security. Consider implementing a minimum length policy.
    """)

    # Lowercase Frequency by Password Strength
    st.write("**Lowercase Frequency by Password Strength**")
    fig_lowercase = px.box(data, x='strength', y='lowercase_freq', title="Lowercase Frequency by Password Strength")
    st.plotly_chart(fig_lowercase)
    st.write("""
    **Insight**: This boxplot shows the distribution of lowercase letter frequency across different password strength categories. Passwords classified as "strong" generally display a balanced use of lowercase letters, often in combination with uppercase letters, digits, and special characters. A lack of variability in lowercase frequency within weak passwords indicates insufficient complexity.
    **Recommendation**: Encourage users to use a mix of uppercase and lowercase letters. Password policies should emphasize the importance of character diversity.
    """)
    
    # Password Length by Strength
    st.write("**Password Length by Strength**")
    fig_violin = px.violin(data, x='strength', y='length', box=True, points="all", title="Password Length by Strength")
    st.plotly_chart(fig_violin)
    st.write("""
    **Insight**: The violin plot highlights that longer passwords are generally associated with higher strength scores. This is consistent with security best practices, where password length is a key determinant of resistance to various attack vectors, such as dictionary and brute-force attacks.
    **Recommendation**: Implement policies that enforce longer passwords as a standard. Regularly educate users on the benefits of longer, more complex passwords.
    """)

    # Special Character Use by Password Length
    st.write("**Special Character Use by Password Length**")
    fig_scatter = px.scatter(data, x='length', y='special_case_freq', color='strength',
                             title="Special Character Use by Password Length")
    st.plotly_chart(fig_scatter)
    st.write("""
    **Insight**: This scatter plot shows the relationship between the number of special characters and password length. Strong passwords tend to include special characters, but this isn't uniformly applied across all length categories.
    **Recommendation**: Encourage users to maintain a consistent use of special characters regardless of password length. This consistency will help strengthen shorter passwords that might otherwise be weaker.
    """)

    # Correlation Heatmap
    st.write("**Correlation Heatmap**")
    corr = data[['length', 'lowercase_freq', 'special_case_freq', 'strength']].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis')
    fig_corr.update_layout(title="Correlation Heatmap")
    st.plotly_chart(fig_corr)
    st.write("""
    **Insight**: This heatmap reveals the correlations between different features and password strength. Features like special character frequency and length show a strong positive correlation with password strength, indicating that these factors are key contributors to a secure password.
    **Recommendation**: Focus on enhancing the use of features that positively correlate with strong passwords. For example, promoting the use of special characters and increasing password length are actionable strategies to improve password strength across the user base.
    """)

    # Password Complexity vs. Strength
    st.write("**Password Complexity vs. Strength**")
    data['complexity'] = data['length'] * (1 + data['special_case_freq']) * (1 + data['lowercase_freq'])
    fig_complexity = px.bar(data, x='strength', y='complexity', title="Password Complexity vs. Strength",
                            color='strength', barmode='group')
    st.plotly_chart(fig_complexity)
    st.write("""
    **Insight**: This bar chart compares the average complexity score across different password strength categories. Strong passwords tend to have higher complexity scores, combining length, character diversity, and special character use.
    **Recommendation**: Implement a complexity score as part of the password creation process. By educating users on the components of complexity (length, diversity, special characters), you can drive the creation of stronger, more secure passwords.
    """)

# Footer with GitHub and LinkedIn links
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: grey;'>
        Made by Paras<br>
        <a href='https://www.linkedin.com/in/parasagrawal1709/' style='color: grey; margin-right: 10px;'>LinkedIn</a> | 
        <a href='https://github.com/parasagrawal1709' style='color: grey;'>GitHub</a>
    </div>
    """, unsafe_allow_html=True)
