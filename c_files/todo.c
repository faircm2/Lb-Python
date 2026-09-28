#include <gtk/gtk.h>
#include <stdio.h>
#include <string.h>

#define FILE_NAME "todo.txt"

typedef struct
{
    GtkWidget *entry;
    GtkWidget *tree;
    GtkListStore *store;
} TodoApp;

/* Add a new task to the list */
static void add_task(GtkWidget *widget, gpointer data)
{
    TodoApp *app = data;
    const char *text;
    GtkTreeIter iter;

    text = gtk_entry_get_text(GTK_ENTRY(app->entry));

    if (strlen(text) == 0)
    {
        return;
    }

    gtk_list_store_append(app->store, &iter);

    gtk_list_store_set(app->store,
                       &iter,
                       0, text,
                       -1);

    gtk_entry_set_text(GTK_ENTRY(app->entry), "");
}

/* Remove the selected task */
static void remove_task(GtkWidget *widget, gpointer data)
{
    TodoApp *app = data;

    GtkTreeSelection *selection;
    GtkTreeModel *model;
    GtkTreeIter iter;

    selection =
        gtk_tree_view_get_selection(GTK_TREE_VIEW(app->tree));

    if (gtk_tree_selection_get_selected(selection,
                                        &model,
                                        &iter))
    {
        gtk_list_store_remove(app->store, &iter);
    }
}

/* Save all tasks to a file */
static void save_tasks(GtkWidget *widget, gpointer data)
{
    TodoApp *app = data;

    FILE *file = fopen(FILE_NAME, "w");

    if (file == NULL)
    {
        printf("Could not save file.\n");
        return;
    }

    GtkTreeIter iter;
    gboolean valid;

    valid = gtk_tree_model_get_iter_first(
        GTK_TREE_MODEL(app->store),
        &iter);

    while (valid)
    {
        gchar *text;

        gtk_tree_model_get(GTK_TREE_MODEL(app->store),
                           &iter,
                           0, &text,
                           -1);

        fprintf(file, "%s\n", text);

        g_free(text);

        valid = gtk_tree_model_iter_next(
            GTK_TREE_MODEL(app->store),
            &iter);
    }

    fclose(file);

    printf("Tasks saved.\n");
}

/* Load saved tasks when the program starts */
static void load_tasks(TodoApp *app)
{
    FILE *file = fopen(FILE_NAME, "r");

    if (file == NULL)
    {
        return;
    }

    char line[256];

    while (fgets(line, sizeof(line), file) != NULL)
    {
        line[strcspn(line, "\n")] = '\0';

        if (strlen(line) > 0)
        {
            GtkTreeIter iter;

            gtk_list_store_append(app->store, &iter);

            gtk_list_store_set(app->store,
                               &iter,
                               0, line,
                               -1);
        }
    }

    fclose(file);
}

int main(int argc, char *argv[])
{
    gtk_init(&argc, &argv);

    TodoApp app;

    GtkWidget *window;
    GtkWidget *main_box;
    GtkWidget *top_box;
    GtkWidget *button_box;

    GtkWidget *add_button;
    GtkWidget *remove_button;
    GtkWidget *save_button;

    GtkCellRenderer *renderer;
    GtkTreeViewColumn *column;

    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);

    gtk_window_set_title(GTK_WINDOW(window),
                         "My To-Do List");

    gtk_window_set_default_size(GTK_WINDOW(window),
                                450, 350);

    gtk_container_set_border_width(
        GTK_CONTAINER(window), 10);

    main_box =
        gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);

    gtk_container_add(GTK_CONTAINER(window),
                      main_box);

    /* Entry and Add button */
    top_box =
        gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);

    gtk_box_pack_start(GTK_BOX(main_box),
                       top_box,
                       FALSE,
                       FALSE,
                       0);

    app.entry = gtk_entry_new();

    gtk_entry_set_placeholder_text(
        GTK_ENTRY(app.entry),
        "Enter a task");

    gtk_box_pack_start(GTK_BOX(top_box),
                       app.entry,
                       TRUE,
                       TRUE,
                       0);

    add_button =
        gtk_button_new_with_label("Add");

    gtk_box_pack_start(GTK_BOX(top_box),
                       add_button,
                       FALSE,
                       FALSE,
                       0);

    /* Data model */
    app.store =
        gtk_list_store_new(1, G_TYPE_STRING);

    /* List view */
    app.tree =
        gtk_tree_view_new_with_model(
            GTK_TREE_MODEL(app.store));

    renderer = gtk_cell_renderer_text_new();

    column =
        gtk_tree_view_column_new_with_attributes(
            "Tasks",
            renderer,
            "text", 0,
            NULL);

    gtk_tree_view_append_column(
        GTK_TREE_VIEW(app.tree),
        column);

    gtk_box_pack_start(GTK_BOX(main_box),
                       app.tree,
                       TRUE,
                       TRUE,
                       0);

    /* Remove and Save buttons */
    button_box =
        gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);

    gtk_box_pack_start(GTK_BOX(main_box),
                       button_box,
                       FALSE,
                       FALSE,
                       0);

    remove_button =
        gtk_button_new_with_label("Remove");

    save_button =
        gtk_button_new_with_label("Save");

    gtk_box_pack_start(GTK_BOX(button_box),
                       remove_button,
                       TRUE,
                       TRUE,
                       0);

    gtk_box_pack_start(GTK_BOX(button_box),
                       save_button,
                       TRUE,
                       TRUE,
                       0);

    /* Signals */
    g_signal_connect(add_button,
                     "clicked",
                     G_CALLBACK(add_task),
                     &app);

    g_signal_connect(remove_button,
                     "clicked",
                     G_CALLBACK(remove_task),
                     &app);

    g_signal_connect(save_button,
                     "clicked",
                     G_CALLBACK(save_tasks),
                     &app);

    g_signal_connect(window,
                     "destroy",
                     G_CALLBACK(gtk_main_quit),
                     NULL);

    /* Load existing todo.txt */
    load_tasks(&app);

    gtk_widget_show_all(window);

    gtk_main();

    g_object_unref(app.store);

    return 0;
}