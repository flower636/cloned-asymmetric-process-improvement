from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import os

TOKEN = os.getenv("BOT_TOKEN")

def start(update, context):
    update.message.reply_text("Hai! Ketik sesuatu, nanti gue kasih link Pinterest.")

def pinterest(update, context):
    query = update.message.text
    link = "https://www.pinterest.com/search/pins/?q=" + query.replace(" ", "%20")
    update.message.reply_text(f"Nih 👉 {link}")

updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher
dp.add_handler(CommandHandler("start", start))
dp.add_handler(MessageHandler(Filters.text & ~Filters.command, pinterest))

updater.start_polling()
updater.idle()